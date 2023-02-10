"""

    Advantage Actor Critic (A2C) with causality algorithm implementation

    Credits: Nicolás Cuadrado, Alejandro Gutierrez, MBZUAI, OptMLLab

"""

import traceback
import numpy as np
import torch
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
from src.utils.cvxpy_own import loop_env, get_all_actions
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)
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

        input = torch.cat([attr, obs], dim=2)
        input = F.relu(self.input(input))

        output = F.softmax(self.output(input), dim=2)

        return output

class Critic(Module):

    def __init__(self, obs_dim, attr_dim, hidden_dim=64) -> None:

        super(Critic, self).__init__()

        self.input = Linear(obs_dim + attr_dim, hidden_dim)
        self.output = Linear(hidden_dim, 1)

    def forward(self, obs, attr):

        input = torch.cat([attr, obs], dim=3)

        output = F.relu(self.input(input))

        output = self.output(output)

        return output

# class Actor(Module):

#     def __init__(self, obs_dim, attr_dim, act_dim, hidden_dim=64) -> None:

#         super(Actor, self).__init__()

#         self.obs_input = Linear(obs_dim, hidden_dim)
#         self.obs_fc = Linear(hidden_dim, hidden_dim*2)
#         self.attr_input = Linear(attr_dim, hidden_dim)
#         self.attr_fc = Linear(hidden_dim, hidden_dim*2)
#         self.concat_fc = Linear(hidden_dim*4, hidden_dim*2)
#         self.output = Linear(hidden_dim*2, act_dim)

#     def forward(self, obs, attr):

#         obs = F.selu(self.obs_input(obs))
#         obs = F.selu(self.obs_fc(obs))

#         att = F.selu(self.attr_input(attr))
#         att = F.selu(self.attr_fc(att))

#         output = torch.cat([att, obs], dim=2)
#         output = F.selu(self.concat_fc(output))

#         output = F.softmax(self.output(output), dim=2)

#         return output

# class Critic(Module):

#     def __init__(self, obs_dim, attr_dim, hidden_dim=64) -> None:

#         super(Critic, self).__init__()

#         self.input_obs = Linear(obs_dim, hidden_dim)
#         self.input_attr = Linear(attr_dim, hidden_dim)
#         self.output = Linear(hidden_dim*2, 1)

#     def forward(self, obs, attr):

#         obs = F.selu(self.input_obs(obs))
#         att = F.relu(self.input_attr(attr))

#         output = torch.cat([att, obs], dim=3)
#         output = self.output(output)

#         return output

# class Actor(Module):

#     def __init__(self, obs_dim, attr_dim, act_dim, hidden_dim=64) -> None:

#         super(Actor, self).__init__()

#         self.obs_input = Linear(obs_dim, hidden_dim)
        
#         self.output = Linear(hidden_dim, act_dim)

#     def forward(self, obs, attr):

#         obs = F.relu(self.obs_input(obs))
        
#         output = F.softmax(self.output(obs), dim=2)

#         return output

# class Critic(Module):

#     def __init__(self, obs_dim, attr_dim, hidden_dim=64) -> None:

#         super(Critic, self).__init__()

#         self.input_obs = Linear(obs_dim, hidden_dim)
        
#         self.output = Linear(hidden_dim, 1)

#     def forward(self, obs, attr):

#         obs = F.relu(self.input_obs(obs))
#         output = self.output(obs)

#         return output

class Agent:

    def __init__(
        self,config, env: Env,  resumed: bool = False 
    ):

        # Get env and its params

        self.env = env
        self.batch_size = config['env']['batch_size']
        self.rollout_steps = config['env']['rollout_steps']
        self.training_steps = config['env']['training_steps']
        self.encoding = config['env']['encoding']
        self.central_agent = config['env']['central_agent']
        self.disable_noise = config['env']['disable_noise']
        
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
        self.min_loss = 0.01

        # Other params 
        self.resumed = resumed
        self.current_step = 0

        '''
            Setup all the configurations for Wandb
        '''

        #TODO review all params are uploaded

        wdb_config={
            "training_steps": self.training_steps,
            "batch_size": self.batch_size,
            "rollout_steps": self.rollout_steps,
            "agent_actor_lr": self.actor_lr,
            "agent_critic_lr": self.critic_lr,
            "agent_actor_nn": self.actor_nn,
            "agent_critic_nn": self.critic_nn,
            "gamma": self.gamma,
            "central_agent": self.central_agent,
            "random_soc_0": env.mg.houses[0].battery.random_soc_0, # It will be the same for all houses
            "encoding": self.encoding,
            "extended_observation": self.extended_observation,
            "num_disc_act": self.num_disc_act,
            "disable_noise": self.disable_noise
        }

        self.discrete_actions = np.linspace(-0.9, 0.9, self.num_disc_act)

        self.wdb_logger = self.setup_wandb_logger(config=wdb_config, tags=["a2c-caus", "discrete"])

        # Enable GPU if available

        if self.enable_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Running on GPU")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        #     print("Running on MPS")
        else:
            self.device = torch.device("cpu")
            print("Running on CPU")
        

        # Configure predictors

        if self.extended_observation:

            print('This model does not support extended observations yet')

            obs_dim = env.obs_size

        else:

            obs_dim = env.obs_size

        attr_dim = env.house_attr_size

        # Configure neural networks

        self.actor = Actor(obs_dim=obs_dim, attr_dim=attr_dim ,act_dim=len(self.discrete_actions), hidden_dim=self.actor_nn).to(self.device)
        self.critic = Critic(obs_dim=obs_dim, attr_dim=attr_dim, hidden_dim=self.critic_nn).to(self.device)

        self.actor.optimizer = Adam(params=self.actor.parameters(), lr=self.actor_lr)
        self.critic.optimizer = Adam(params=self.critic.parameters(), lr=self.critic_lr)

        # Check if we are resuming training from a previous checkpoint

        if resumed:

            checkpoint = torch.load(self.wdb_logger.load_model().name)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor.optimizer.load_state_dict(checkpoint['actor_opt_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic.optimizer.load_state_dict(checkpoint['critic_opt_state_dict'])
            self.current_step = checkpoint['current_step']

        self.actor.train()
        self.critic.train()

        # Hooks into the models to collect gradients and topology

        self.wdb_logger.watch_model(models=(self.actor, self.critic))

    def setup_wandb_logger(self, config: dict, tags: list):
        
        wdb_logger = WandbLogger(tags=tags)

        wdb_logger.disable_logging(disable=self.disable_logging)

        wdb_logger.init(config=config)

        return wdb_logger

    def select_action(self, state: Tensor):

        probs = self.actor(obs=state, attr=tensor(self.env.houses_attr, device=self.device).float())

        # Define the distribution

        dist = Categorical(probs=probs)

        # Sample action 

        action_index = dist.sample()
        action = np.stack([[[self.discrete_actions[batch_index.cpu()]] for batch_index in batch_actions] for batch_actions in action_index])

        log_prob = dist.log_prob(action_index)

        return action, log_prob

    def get_extended_observations(self, state):

        print('This model does not support extended observations yet')

        return state

    def rollout(self):

        states, rewards, log_probs, actions_hist = [], [], [], []

        # Get the initial state by resetting the environment

        state, reward, done, _ = self.env.reset()

        if self.extended_observation:

            state = self.get_extended_observations(state)

        # Launch rollout

        while not done:

            # Start by appending the state to create the states trajectory
            states.append(state)

            # Perform action and pass to next state

            actions, log_prob = self.select_action(tensor(state).float().to(self.device))

            state, reward, done, _ = self.env.step(actions)

            if self.extended_observation:

                state = self.get_extended_observations(state)

            rewards.append(reward)
            log_probs.append(log_prob)
            actions_hist.append(actions)

        rewards = np.stack(rewards)
        actions_hist = np.stack(actions_hist).squeeze(axis=-1)

        return states, rewards, log_probs, actions_hist

    def check_model(self, mode: str = 'eval'):

        # Change environment to evaluation mode

        self.env.change_mode(mode=mode)
        self.actor.eval()
        self.critic.eval()

        # Evaluate current model

        with torch.no_grad():

            self.rollout()

        price_metric, emission_metric = self.env.mg.get_houses_metrics()

        # Change environment back to training mode

        self.env.change_mode(mode='train')
        self.actor.train()
        self.critic.train()

        return price_metric, emission_metric

    def train(self):

        # Rollout registers
        
        all_states, all_rewards, all_actions, all_net_energy = [], [], [], []

        # Metrics registers

        train_price_metric, train_emission_metric, eval_price_metric, eval_emission_metric = [], [], [], []

        for step in tqdm(range(self.current_step, self.training_steps)):

            # Perform rollouts and sample trajectories

            states, rewards, log_probs, actions_hist = self.rollout()
            
            # Append the trajectories to the arrays

            all_states.append(states)
            all_rewards.append(rewards)
            all_actions.append(actions_hist)
            all_net_energy.append(self.env.mg.net_energy)

            # Perform the optimization step

            log_probs = torch.stack(log_probs, dim=0)

            states = tensor(np.array(states)).float().to(self.device)
            houses_attr = tensor(np.repeat(self.env.houses_attr[np.newaxis,:,:,:], self.rollout_steps, axis=0), device=self.device).float()
            value = self.critic(obs=states, attr=houses_attr).squeeze(dim=-1)

            # Causality trick considering gamma

            sum_rewards = []
            prev_reward = 0

            for reward in reversed(rewards):
                prev_reward = np.copy(reward + self.gamma * prev_reward)
                sum_rewards.insert(0, prev_reward+0.0)

            sum_rewards = tensor(np.stack(sum_rewards)).squeeze(dim=-1).float().to(self.device)

            # Perform the optimization step

            try:

                actor_loss = - torch.mean(torch.sum(log_probs*(sum_rewards - value.detach()), dim=0))
                critic_loss = MSELoss()(value, sum_rewards)

                # Backpropagation to train Actor NN

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                # Backpropagation to train Critic NN

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

            except Exception as e:

                traceback.print_exc()

            # Log the metrics

            t_price_metric, t_emission_metric = self.env.mg.get_houses_metrics()

            train_price_metric.append(t_price_metric.mean())
            train_emission_metric.append(t_emission_metric.mean())

            # Evaluate the model

            e_price_metric, e_emission_metric = self.check_model()

            eval_price_metric.append(e_price_metric.mean())
            eval_emission_metric.append(e_emission_metric.mean())

            # Check stop condition

            stop_condition = actor_loss.abs().item() <= self.min_loss and critic_loss.abs().item() <= self.min_loss
            
            if step % 50 == 0 or stop_condition:

                # Wandb logging

                results = {
                    # Mean from batches, sum from rollout steps (24) and mean houses
                    "rollout_avg_reward": rewards.mean(axis=2).sum(axis=0).mean(),
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "avg_action": actions_hist.mean(),
                }

                self.wdb_logger.log_dict(results)

            if step % 250 == 0:

                # Save networks weights for resume training

                self.save_weights(
                    actor_state_dict=self.actor.state_dict(),
                    actor_opt_state_dict=self.actor.optimizer.state_dict(),
                    critic_state_dict=self.critic.state_dict(),
                    critic_opt_state_dict=self.critic.optimizer.state_dict(),
                    current_step=step
                )

                self.wdb_logger.save_model()

        # Return results dictionary
        return {
            "rollout_avg_reward": rewards.mean(axis=2).mean(axis=1).sum(),
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
                "cvxpy":{
                    "price_metric": train_price_metric * 0,
                    "emission_metric": train_emission_metric * 0
                },
            },
            "eval": {
                "agent":{
                    "price_metric": eval_price_metric,
                    "emission_metric": eval_emission_metric
                },
                "cvxpy":{
                    "price_metric": train_price_metric * 0,
                    "emission_metric": eval_emission_metric * 0
                },
            },
            "test": {
            },
        }

    def test(self):

        test_price_metric, test_emission_metric = [], []

        for step in tqdm(range(self.current_step, self.training_steps)):

            # Evaluate the model

            e_price_metric, e_emission_metric = self.check_model(mode='test')

            test_price_metric.append(e_price_metric.mean())
            test_emission_metric.append(e_emission_metric.mean())
            
        return { "agent":{
                "price_metric": test_price_metric,
                "emission_metric": test_emission_metric
            },"cvxpy":{
                    "price_metric": test_price_metric * 0,
                    "emission_metric": test_emission_metric * 0
                },
        }

    # Save weights to file

    def save_weights(self, actor_state_dict, actor_opt_state_dict, critic_state_dict, critic_opt_state_dict, current_step):

        model_path = self.wdb_logger.run.dir if self.wdb_logger.run is not None else './models'

        torch.save({
            'current_step': current_step,
            'actor_state_dict': actor_state_dict,
            'actor_opt_state_dict': actor_opt_state_dict,
            'critic_state_dict': critic_state_dict,
            'critic_opt_state_dict': critic_opt_state_dict,
        }, f'{model_path}/2h_d_a2c"_model.pt')

        print(f'Saving model on step: {current_step}')


"""
    Main method definition
"""

if __name__ == '__main__':
    model = "d_a2c_mgE1"

    config = load_config(model)
    config = config['train']
    
    # Start wandb logger

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

        results['test'] = agent.test()

        # Make plots
        #CVXPY
        env = SimpleMicrogrid(config=config['env'])
            # Train
        mode = 'train'
        rewards_t, battery_values_t, action_values_t = get_all_actions(env, mode)
        rewards_t_env,train_metrics = loop_env(env, action_values_t, mode)
        results['train']['cvxpy']['price_metric'] = [train_metrics[0].mean() ] * 2000
        results['train']['cvxpy']['emission_metric'] = [train_metrics[1].mean() ] * 2000

            # Eval
        mode = 'eval'
        rewards_e, battery_values_e, action_values_e = get_all_actions(env, mode)
        rewards_e_env, eval_metrics = loop_env(env, action_values_e, mode)
        results['eval']['cvxpy']['price_metric'] = [eval_metrics[0].mean() ] * 2000
        results['eval']['cvxpy']['emission_metric'] = [eval_metrics[1].mean() ] * 2000

            # Test
        mode = 'test'
        rewards_s, battery_values_s, action_values_S = get_all_actions(env, mode)
        rewards_s_env, test_metrics = loop_env(env, action_values_S, mode)
        results['test']['cvxpy']['price_metric'] = [test_metrics[0].mean() ] * 2000
        results['test']['cvxpy']['emission_metric'] = [test_metrics[1].mean() ] * 2000


        plot_metrics(metrics=results)

        # plot_rollout(env=my_env, results=results)
        
        # Finish wandb process

        agent.wdb_logger.finish()

    except (RuntimeError, KeyboardInterrupt):

        traceback.print_exc()