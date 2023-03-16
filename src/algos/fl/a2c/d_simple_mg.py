"""

    Advantage Actor Critic (A2C) with causality algorithm implementation

    Credits: Nicolás Cuadrado, Alejandro Gutierrez, MBZUAI, OptMLLab

"""
import traceback
import numpy as np
import torch
import argparse
import pickle

from tqdm import tqdm
from gym import Env
from torch import Tensor, tensor
from torch.nn import Module, Linear, MSELoss, Flatten
from torch.functional import F
from torch.optim import Adam
from torch.distributions import Categorical

from src.utils.wandb_logger import WandbLogger
from src.environments.simple_microgrid import SimpleMicrogrid

from src.utils.tools import set_all_seeds, load_config, plot_rollout, plot_metrics
torch.autograd.set_detect_anomaly(True)

# Define global variables

ZERO = 1e-5

'''
    Agent definitions
'''

class Actor(Module):

    def __init__(self, obs_dim, attr_dim, act_dim, hidden_dim=64) -> None:

        super(Actor, self).__init__()

        self.input = Linear(obs_dim + attr_dim, hidden_dim)
        self.output = Linear(hidden_dim, act_dim)

    def forward(self, obs, attr):

        input = torch.cat([attr, obs], dim=1)
        input = F.relu(self.input(input))

        output = F.softmax(self.output(input), dim=1)

        return output

class Critic(Module):

    def __init__(self, obs_dim, attr_dim, hidden_dim=64) -> None:

        super(Critic, self).__init__()

        self.input = Linear(obs_dim + attr_dim, hidden_dim)
        self.output = Linear(hidden_dim, 1)

    def forward(self, obs, attr):

        input = torch.cat([attr, obs], dim=2)

        output = F.relu(self.input(input))

        output = self.output(output)

        return output

class Agent:

    def __init__(
        self,config, env: Env,  resumed: bool = False 
    ):

        # Get env and its params
        self.counter = 0

        self.env = env
        self.batch_size = config['env']['batch_size']
        self.rollout_steps = config['env']['rollout_steps']
        self.switch_steps = config['env']['switch_steps']
        self.training_steps = config['env']['training_steps']
        self.encoding = config['env']['encoding']
        self.central_agent = config['env']['central_agent']
        self.disable_noise = config['env']['disable_noise']

        # Federated Learning Params
        self.sync_steps = config['env']['sync_steps']
        
        config = config['agent']

        # Get params from yaml config file

        self.num_disc_act = config['num_disc_act']
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.actor_nn = config['actor_nn']
        self.critic_nn = config['critic_nn']
        self.gamma = config['gamma']
        self.disable_logging = config['disable_logging']
        self.enable_gpu = config['enable_gpu']
        self.extended_observation = config['extended_observation']
        # self.early_stop = config['early_stop'] #TODO review this name and min loss
        self.min_loss = config['min_loss_stop_condition']

        # Other params 
        self.resumed = resumed
        self.current_step = 0

        '''
            Setup all the configurations for Wandb
        '''
        wdb_config={
            "training_steps": self.training_steps,
            "batch_size": self.batch_size,
            "rollout_steps": self.rollout_steps,
            "agent_actor_lr": self.actor_lr,
            "agent_critic_lr": self.critic_lr,
            "agent_actor_nn": self.actor_nn,
            "agent_critic_nn": self.critic_nn,
            "gamma": self.gamma,
            "sync_steps": self.sync_steps,
            "central_agent": self.central_agent,
            "random_soc_0": env.mg.houses[0].battery.random_soc_0, # It will be the same for all houses
            "encoding": self.encoding,
            "extended_observation": self.extended_observation,
            "num_disc_act": self.num_disc_act,
            "disable_noise": self.disable_noise
        }

        self.wdb_logger = self.setup_wandb_logger(config=wdb_config, tags=["a2c-caus", "discrete", "fedavg"])

        # Define discrete actions

        self.discrete_actions = np.linspace(-0.9, 0.9, self.num_disc_act)

        # Enable GPU if available

        if self.enable_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Running on GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on CPU")

        # Configure predictors
        if self.extended_observation:
            print('This model does not support extended observations yet')

            self.obs_dim = env.obs_size
        else:
            self.obs_dim = env.obs_size

        self.attr_dim = env.attr_size

        # Configure neural networks, List of NN 
        self.init_actors_critics()

        # Configure oweightsptimizers
        for actor in self.actor_list:
            actor.train()
    
        for critic in self.critic_list:
            critic.train()

        # Checkpoint load from Wandb from Global model 
        if resumed:
            self.load_checkpoint()
        # Hooks into the models global to collect gradients and topology
        self.wdb_logger.watch_model(models=(self.actor_list[0], self.critic_list[0]))

    def setup_wandb_logger(self, config: dict, tags: list):
        
        wdb_logger = WandbLogger(tags=tags)

        wdb_logger.disable_logging(disable=self.disable_logging)

        wdb_logger.init(config=config)

        return wdb_logger
    
    # Initialize the actor and critic networks
    def init_actors_critics(self):
        self.actor_list = [Actor(obs_dim=self.obs_dim, attr_dim=self.attr_dim ,act_dim=len(self.discrete_actions), hidden_dim=self.actor_nn).to(self.device) for _ in range(self.env.n_houses)]
        self.critic_list = [Critic(obs_dim=self.obs_dim, attr_dim=self.attr_dim, hidden_dim=self.critic_nn).to(self.device) for _ in range(self.env.n_houses)]
        
        # Configure optimizers
        for actor in self.actor_list:
            actor.optimizer = Adam(params=actor.parameters(), lr=self.actor_lr)
    
        for critic in self.critic_list:
            critic.optimizer = Adam(params=critic.parameters(), lr=self.critic_lr) 


    # Main training loop 
    def train(self):

        # Rollout registers
    
        all_states, all_rewards, all_actions, all_net_energy = [], [], [], []

        # Metrics registers

        train_price_metric, train_emission_metric, eval_price_metric, eval_emission_metric = [], [], [], []

        # Losses history 

        actor_loss_hist, critic_loss_hist =  np.empty(shape=(self.env.n_houses, 1)),  np.empty(shape=(self.env.n_houses, 1))

        # Loop through all the training steps = epochs = episodes
        for step in tqdm(range(self.current_step, self.training_steps)):

            # Perform rollouts and sample trajectories

            states, rewards, log_probs, actions_hist, actions_ix_hist, actions_probs = self.rollout()
            
            # Append the trajectories to the registers
            # all_states.append(states)
            # all_rewards.append(rewards)
            # all_actions.append(actions_hist)
            # all_net_energy.append(self.env.mg.net_energy)

            # Move states to tensor and to device
            states = tensor(np.array(states)).float().to(self.device)
           
            # Causality trick considering gamma

            sum_rewards = []
            prev_reward = 0

            for reward in reversed(rewards):
                prev_reward = np.copy(reward + self.gamma * prev_reward)
                sum_rewards.insert(0, prev_reward+0.0)

            sum_rewards = tensor(np.stack(sum_rewards)).squeeze(dim=-1).float().to(self.device)

            # Loop through all the actors and train them
            for i, (actor, critic) in enumerate(zip(self.actor_list, self.critic_list)):
                try:
                    # Get the log prob of the actor 
                    agent_log_probs = torch.stack([p[i] for p in log_probs])
                    # Get actions
                    # agent_action_ix = torch.tensor(actions_ix_hist[:, i, :].reshape(self.rollout_steps, 1, self.batch_size)).to(self.device)

                    # Modify house attr matrix to fit number of time steps for critic input
                    attr = tensor(np.repeat(self.env.attr[np.newaxis,i,:,:], self.rollout_steps, axis=0), device=self.device).float()
                   
                    # Get the value of the critic, send only state of 1 house
                    value = critic(states[:,i,:,:], attr)

                    # Drop the last dimension of the value
                    value = value.squeeze(dim=-1)

                    # Get the loss of the specific actor and critic
                    actor_loss = - torch.mean(torch.sum(agent_log_probs*(sum_rewards[:,i,:] - value.detach()), dim=0))
                    critic_loss = MSELoss()(value, sum_rewards[:,i,:])

                    # Separate and save the losses per house
                    # actor_loss_hist[i] = np.append(actor_loss_hist[i], [actor_loss.item()])
                    # critic_loss_hist[i] = np.append(critic_loss_hist[i], critic_loss.item())

                    # Backpropagation to train single Actor NN

                    actor.optimizer.zero_grad()
                    actor_loss.backward()
                    actor.optimizer.step()

                    # Backpropagation to train single Critic NN

                    critic.optimizer.zero_grad()
                    critic_loss.backward()
                    critic.optimizer.step()

                except Exception as e:

                    traceback.print_exc()

            # Log the metrics

            t_price_metric, t_emission_metric = self.env.mg.get_houses_metrics()

            train_price_metric.append(t_price_metric.mean())
            train_emission_metric.append(t_emission_metric.mean())

            # Evaluate the model
            self.env.change_mode(mode='eval')
            e_price_metric, e_emission_metric = self.check_model()

            eval_price_metric.append(e_price_metric.mean())
            eval_emission_metric.append(e_emission_metric.mean())

            # Update Actor and Critic FL weights
            if step % self.sync_steps == 0:
                self.sync_weights()
                print("el federated learning")

                # Save global weights for resume training
                if step % self.sync_steps == 0:

                    # Save networks weights for resume training

                    self.save_weights(
                        actor_state_dict=self.actor_list[0].state_dict(),
                        actor_opt_state_dict=self.actor_list[0].optimizer.state_dict(),
                        critic_state_dict=self.critic_list[0].state_dict(),
                        critic_opt_state_dict=self.critic_list[0].optimizer.state_dict(),
                        current_step=step
                    )

                    self.wdb_logger.save_model()

            # Rotate grid profile after each episode

            if step != 0 and step % self.switch_steps == 0:

                self.env.mg.change_grid_profile()

            # Check stop condition

            stop_condition = actor_loss.abs().item() <= self.min_loss and critic_loss.abs().item() <= self.min_loss
            
            # Log results in wandb
            if step != 0 and step % 50 == 0 or stop_condition:
                self.counter = self.counter + 1

                # Wandb logging
                results = {
                    "train_price_metric": t_price_metric.mean(),
                    "train_emission_metric": t_emission_metric.mean(),
                    "rollout_avg_reward": rewards.mean(axis=2).sum(axis=0).mean(),
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "avg_action": actions_hist.mean(),
                }

                self.wdb_logger.log_dict(results)

        # Return results dictionary
        return {
            "training_steps": self.training_steps,
            "rollout_steps": self.rollout_steps,
            "train": {
                "agent":{
                    "price_metric": train_price_metric,
                    "emission_metric": train_emission_metric,
                    "states": all_states,
                    "rewards": all_rewards,
                    "actions": all_actions,
                    "net_energy": all_net_energy
                },
                "actor_loss": actor_loss_hist,
                "critic_loss": critic_loss_hist
            },
            "eval": {
                "agent":{
                    "price_metric": eval_price_metric,
                    "emission_metric": eval_emission_metric
                },
            },
        }
 
    def select_action(self, state: Tensor):

        actions = []
        agents_log_probs = []
        indexes = []
        actions_probs = []

        for i, actor in enumerate(self.actor_list):

            probs = actor(obs=state[i,:,:], attr=tensor(self.env.attr[i], device=self.device).float())

            # Define the distribution

            dist = Categorical(probs=probs)

            # Sample action 

            action_index = dist.sample()
            action = self.discrete_actions[action_index.cpu()]

            log_prob = dist.log_prob(action_index)

            actions.append(action)
            indexes.append(action_index.cpu())
            agents_log_probs.append(log_prob)
            actions_probs.append(probs)

        # Add batch dimension
        actions = np.expand_dims(np.stack(actions), axis=2)
        indexes = np.expand_dims(np.stack(indexes), axis=2)
        actions_probs = torch.stack(actions_probs)

        return actions, agents_log_probs, indexes, actions_probs

    def get_extended_observations(self, state):

        print('This model does not support extended observations yet')

        return state

# Goes through the environment and collects the states, rewards and actions taken, one trajectory at a time. 1 trajectory = 1 rollout
    def rollout(self):

        states, rewards, log_probs, actions_hist, actions_ix_hist, actions_probs = [], [], [], [], [], []

        # Get the initial state by resetting the environment

        state, reward, done, _ = self.env.reset()

        if self.extended_observation:

            state = self.get_extended_observations(state)

        # Launch rollout

        while not done:

            # Start by appending the state to create the states trajectory
            states.append(state)

            # Perform action and pass to next state

            actions, log_prob, indexes, probs = self.select_action(tensor(state).float().to(self.device))

            state, reward, done, _ = self.env.step(actions)

            if self.extended_observation:
                state = self.get_extended_observations(state)

            rewards.append(reward)
            log_probs.append(log_prob)
            actions_hist.append(actions)
            actions_ix_hist.append(indexes)
            actions_probs.append(probs)

        rewards = np.stack(rewards)
        actions_hist = np.stack(actions_hist).squeeze(axis=-1)
        actions_ix_hist = np.stack(actions_ix_hist).squeeze(axis=-1)
        actions_probs = torch.stack(actions_probs)

        return states, rewards, log_probs, actions_hist, actions_ix_hist, actions_probs

    # Evaluation pipeline for the model
    def check_model(self):
        # Freeze the weights of the model
        for actor in self.actor_list:
            actor.eval()
    
        for critic in self.critic_list:
            critic.eval()

        # Evaluate current model

        with torch.no_grad():

            self.rollout()

        price_metric, emission_metric = self.env.mg.get_houses_metrics()

        # Change environment back to training mode

        self.env.change_mode(mode='train')
        for actor in self.actor_list:
            actor.train()
    
        for critic in self.critic_list:
            critic.train()

        return price_metric, emission_metric
    
    # Test the model with new grid profiles
    def test(self):

        test_price_metric, test_emission_metric = [], []

        for step in tqdm(range(self.current_step, self.training_steps)):

            # Change environment to evaluation mode
            
            self.env.change_mode(mode='test')

            # Create new actors and critics for the new environment and load the weights of the global model
            self.init_actors_critics()
            # Load the weights of the global model, before freezing them in the check_model evaluation
            self.load_checkpoint()

            e_price_metric, e_emission_metric = self.check_model()

            test_price_metric.append(e_price_metric.mean())
            test_emission_metric.append(e_emission_metric.mean())

            if step != 0 and step % self.switch_steps == 0:

                self.env.mg.change_grid_profile()
            
        return { 
            "agent":{
                "price_metric": test_price_metric,
                "emission_metric": test_emission_metric
            },
        }
   

    # Federated Learning step to sync weights between all the actors and all the critics
    def sync_weights(self):
        # ---- ACTOR
        # Sync weights between the local and global models, FedAvg
        state_dict = self.actor_list[0].state_dict()
        mean_model_actor = {}

        # Get the mean of the weights of all the actors
        for layer in state_dict.keys():
            param_avg = torch.stack([self.actor_list[i].state_dict()[layer] for i in range(self.env.n_houses)], axis=0).mean(axis=0)
            mean_model_actor[layer] = param_avg.clone()
        
        # Update the global model with the mean of the local models
        for i, actor in enumerate(self.actor_list):
            self.actor_list[i].load_state_dict(mean_model_actor)

        # ---- CRITIC
        # Sync weights between the local and global models, FedAvg
        state_dict = self.critic_list[0].state_dict()
        mean_model_critic = {}

        # Get the mean of the weights of all the actors
        for layer in state_dict.keys():
            param_avg = torch.stack([self.critic_list[i].state_dict()[layer] for i in range(self.env.n_houses)], axis=0).mean(axis=0)
            mean_model_critic[layer] = param_avg.clone()
        
        # Update the global model with the mean of the local models
        for i, critic in enumerate(self.critic_list):
            self.critic_list[i].load_state_dict(mean_model_critic)
    
    # Load the parameters from the global model
    def load_checkpoint(self):
        # Load parameters from wand logger or fallback to local file
        model_path = self.wdb_logger.run.dir if self.wdb_logger.run is not None else './models/fl'
        filename =f"sync_steps_{config['env']['sync_steps']}_alr_{config['agent']['actor_lr']}_clr_{config['agent']['critic_lr']}_cnn_{config['agent']['actor_nn']}_ann_{config['agent']['critic_nn']}"

        checkpoint = torch.load(f'{model_path}/{filename}.pt')
        # if self.wdb_logger.run is not None:
        #     checkpoint = torch.load('./models/2h_d_a2c_mg_fl_model.pt')
        # else:
        #     checkpoint = torch.load(self.wdb_logger.load_model().name)

        for i, (actor, critic) in enumerate(zip(self.actor_list, self.critic_list)):
            actor.load_state_dict(checkpoint['actor_state_dict'])
            actor.optimizer.load_state_dict(checkpoint['actor_opt_state_dict'])
            critic.load_state_dict(checkpoint['critic_state_dict'])
            critic.optimizer.load_state_dict(checkpoint['critic_opt_state_dict'])

        self.current_step = checkpoint['current_step']
    
    # Save weights to file or upload to wandb
    def save_weights(self, actor_state_dict, actor_opt_state_dict, critic_state_dict, critic_opt_state_dict, current_step):

        model_path = self.wdb_logger.run.dir if self.wdb_logger.run is not None else './models/fl'
        filename =f"sync_steps_{config['env']['sync_steps']}_alr_{config['agent']['actor_lr']}_clr_{config['agent']['critic_lr']}_cnn_{config['agent']['actor_nn']}_ann_{config['agent']['critic_nn']}"

        torch.save({
            'current_step': current_step,
            'actor_state_dict': actor_state_dict,
            'actor_opt_state_dict': actor_opt_state_dict,
            'critic_state_dict': critic_state_dict,
            'critic_opt_state_dict': critic_opt_state_dict,
        }, f'{model_path}/{filename}.pt')

        print(f'Saving model on step: {current_step}')


"""
    Main method definition
"""
# Read arguments from command line

parser = argparse.ArgumentParser(prog='rl', description='RL Experiments')

parser.add_argument("-alr", "--actor_lr", type=float, help="Actor learning rate")
parser.add_argument("-clr", "--critic_lr", type=float, help="Critic learning rate")
parser.add_argument("-ann", "--actor_nn", type=int, help="Actor neurons number")
parser.add_argument("-cnn", "--critic_nn", type=int, help="Critic neurons number")
parser.add_argument("-ss", "--sync_steps", type=int, help="FL sync steps")
parser.add_argument("-f", "--filename", type=str, help="File name")

args = parser.parse_args()

if __name__ == '__main__':
    model = "d_a2c_mg_fl"

    config = load_config(model)
    
    # Parameters to override the config file
    if args.actor_lr is not None:
        config['agent']['actor_lr'] = args.actor_lr
    
    if args.critic_lr is not None:
        config['agent']['critic_lr'] = args.critic_lr

    if args.actor_nn is not None:
        config['agent']['actor_nn'] = args.actor_nn
    
    if args.critic_nn is not None:
        config['agent']['critic_nn'] = args.critic_nn

    if args.sync_steps is not None:
        config['env']['sync_steps'] = args.sync_steps

        # Get arguments and override config with command line arguments

    filename =f"sync_steps_{args.sync_steps}_alr_{args.actor_lr}_clr_{args.critic_lr}_cnn_{args.actor_nn}_ann_{args.critic_nn}"

    try:
        '''
            Run the simulator
        '''

        set_all_seeds(0)

        # Instantiate the environment

        my_env = SimpleMicrogrid(config=config['env'])

        # Instantiate the agent

        agent = Agent(
            env=my_env, config = config
        )

        # Launch the training

        results = agent.train()

        # Check the final model with the test dataset and retrieve metrics

        results['test'] = {}
        results['test'] = agent.test()

        # Save results to pickle file

        results_to_dump = results.copy()

        del results_to_dump['train']['agent']['states']
        del results_to_dump['train']['agent']['rewards']
        del results_to_dump['train']['agent']['actions']
        del results_to_dump['train']['agent']['net_energy']
        with open(f'./results/fl/{model}_{filename}.pkl', 'wb') as f:
            pickle.dump(results_to_dump, f, pickle.HIGHEST_PROTOCOL)

        # Make plots

        # plot_rollout(env=my_env, results=results)
        
        # Finish wandb process 
        print("train", results['train']['agent']['price_metric'][-1], results['train']['agent']['emission_metric'][-1])
        print("test", results['test']['agent']['price_metric'][-1], results['test']['agent']['emission_metric'][-1])

        # plot_metrics(results)
        print(agent.counter)
        agent.wdb_logger.finish()

    except (RuntimeError, KeyboardInterrupt):

        traceback.print_exc()