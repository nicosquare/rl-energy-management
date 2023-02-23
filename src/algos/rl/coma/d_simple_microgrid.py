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
from torch.nn import Module, Linear, MSELoss, GRUCell
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

class Memory:
    def __init__(self, agent_num, action_dim):
        self.agent_num = agent_num
        self.action_dim = action_dim

        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(agent_num)]
        self.reward = []
        self.done = [[] for _ in range(agent_num)]

    def get(self):
        actions = torch.tensor(self.actions)
        observations = self.observations

        pi = []
        for i in range(self.agent_num):
            pi.append(torch.cat(self.pi[i]).view(len(self.pi[i]), self.action_dim))

        reward = torch.tensor(self.reward)
        done = self.done

        return actions, observations, pi, reward, done

    def clear(self):
        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(self.agent_num)]
        self.reward = []
        self.done = [[] for _ in range(self.agent_num)]

class Actor(Module):

    def __init__(self, obs_dim, hidden_dim, act_dim, batch_size):
        super(Actor, self).__init__()

        self.fc1 = Linear(obs_dim, hidden_dim)
        self.rnn = GRUCell(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, act_dim)

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.hidden_state = self.init_hidden()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(self.batch_size, self.hidden_dim).zero_()

    def forward(self, obs):

        obs = F.relu(self.fc1(obs))
        h_in = self.hidden_state.to(obs.device)
        h = self.rnn(obs, h_in)

        self.hidden_state = h.clone()

        q = F.softmax(self.fc2(h), dim=1)

        return q, h

class Critic(Module):

    def __init__(self, obs_dim, agent_num, action_dim, hidden_dim: int = 64):

        super(Critic, self).__init__()

        input_dim = 1 + obs_dim * agent_num + agent_num

        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3 = Linear(hidden_dim, action_dim)

    def forward(self, obs):

        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))

        return self.fc3(obs)

class Agent:

    def __init__(
        self,config, env: Env,  resumed: bool = False 
    ):

        # Get env and its params

        self.env = env
        self.batch_size = config['env']['batch_size']
        self.rollout_steps = config['env']['rollout_steps']
        self.training_steps = config['env']['training_steps']
        self.target_update_steps = config['env']['target_update_steps']
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

        # Configire memory

        self.memory = Memory(env.n_houses, 1)

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

        self.wdb_logger = self.setup_wandb_logger(config=wdb_config, tags=["coma", "discrete"])

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

        # Configure neural networks

        self.actors = [
            Actor(obs_dim=self.obs_dim, act_dim=self.num_disc_act, hidden_dim=self.actor_nn, batch_size=self.batch_size).to(self.device)
            for _ in range(env.n_houses)
        ]
        self.critic = Critic(obs_dim=self.obs_dim, agent_num=env.n_houses, action_dim=self.num_disc_act, hidden_dim=self.critic_nn).to(self.device)

        self.critic_target = Critic(obs_dim=self.obs_dim, agent_num=env.n_houses, action_dim=self.num_disc_act, hidden_dim=self.critic_nn).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        for actor in self.actors:
            actor.optimizer = Adam(params=actor.parameters(), lr=self.actor_lr) 
            actor.train()

        self.critic.train()
        self.critic.optimizer = Adam(params=self.critic.parameters(), lr=self.critic_lr)

        # Check if we are resuming training from a previous checkpoint

        if resumed:

            # TODO: Resuming logic for multiple agents

            checkpoint = torch.load(self.wdb_logger.load_model().name)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor.optimizer.load_state_dict(checkpoint['actor_opt_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic.optimizer.load_state_dict(checkpoint['critic_opt_state_dict'])
            self.current_step = checkpoint['current_step']


        # Hooks into the models to collect gradients and topology

        self.wdb_logger.watch_model(models=(self.actors, self.critic))

    def setup_wandb_logger(self, config: dict, tags: list):
        
        wdb_logger = WandbLogger(tags=tags)

        wdb_logger.disable_logging(disable=self.disable_logging)

        wdb_logger.init(config=config)

        return wdb_logger

    def select_action(self, state: Tensor):

        actions = []
        agents_log_probs = []
        indexes = []
        actions_probs = []

        for ix, actor in enumerate(self.actors):

            probs, _ = actor(obs=state[ix,:,:])

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

        actions = np.expand_dims(np.stack(actions), axis=2)
        indexes = np.expand_dims(np.stack(indexes), axis=2)
        actions_probs = torch.stack(actions_probs)

        return actions, agents_log_probs, indexes, actions_probs

    def get_extended_observations(self, state):

        print('This model does not support extended observations yet')

        return state

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

    def check_model(self, mode: str = 'eval'):

        # Change environment to evaluation mode

        self.env.change_mode(mode=mode)
        
        for actor in self.actors:
            actor.eval()
        self.critic.eval()

        # Evaluate current model

        with torch.no_grad():

            self.rollout()

        price_metric, emission_metric = self.env.mg.get_houses_metrics()

        # Change environment back to training mode

        self.env.change_mode(mode='train')
        
        for actor in self.actors:
            actor.train()
        self.critic.train()

        return price_metric, emission_metric

    def build_input_critic(self, agent_id, observations, actions):

        actions = torch.from_numpy(actions).float().to(self.device).view(self.rollout_steps, self.batch_size, self.env.n_houses)
        observations = observations.view(self.rollout_steps, self.batch_size, self.env.n_houses * self.obs_dim)
        ids = torch.ones(self.rollout_steps, self.batch_size, 1).to(self.device) * agent_id

        input_critic = torch.cat([ids, observations, actions], axis=2)

        return input_critic

    def train(self):

        # Rollout registers
        
        all_states, all_rewards, all_actions, all_net_energy = [], [], [], []

        # Metrics registers

        train_price_metric, train_emission_metric, eval_price_metric, eval_emission_metric = [], [], [], []
        actors_losses, critic_losses = [], []

        for step in tqdm(range(self.current_step, self.training_steps)):

            actors_loss = []

            # Perform rollouts and sample trajectories

            states, rewards, log_probs, actions_hist, actions_ix_hist, actions_probs = self.rollout()
            
            # Append the trajectories to the arrays

            all_states.append(states)
            all_rewards.append(rewards)
            all_actions.append(actions_hist)
            all_net_energy.append(self.env.mg.net_energy)

            states = tensor(np.array(states)).float().to(self.device)

            # Causality trick considering gamma

            sum_rewards = []
            prev_reward = 0

            for reward in reversed(rewards):
                prev_reward = np.copy(reward + self.gamma * prev_reward)
                sum_rewards.insert(0, prev_reward+0.0)

            sum_rewards = tensor(np.stack(sum_rewards)).squeeze(dim=-1).float().to(self.device)

            # Train the actors with each counterfactual critic

            for i, actor in enumerate(self.actors):

                agent_log_probs = torch.stack([p[i] for p in log_probs])

                input_critic = self.build_input_critic(i, states, actions_hist)
                q_target = self.critic_target(obs=input_critic)

                agent_action_ix = torch.tensor(actions_ix_hist[:, i, :].reshape(self.rollout_steps, 1, self.batch_size)).to(self.device)

                q_target_agent = torch.gather(q_target, dim=2, index=agent_action_ix)
                baseline = (actions_probs[:,i,:,:]*q_target).sum().detach()
                advantage = (q_target_agent - baseline).detach()

                actor_loss = - torch.mean(torch.mean(agent_log_probs * advantage.squeeze(), dim=0))

                actor.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 5)
                actor.optimizer.step()

                actors_loss.append(actor_loss.item())

                # Train the critic

                q = self.critic(obs=input_critic)
                q_agent = torch.gather(q, dim=2, index=agent_action_ix)
       
                critic_loss = torch.mean((q_agent.squeeze() - sum_rewards[:,i,:])**2)

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
                self.critic.optimizer.step()

                for actor in self.actors:
                    actor.hidden = actor.init_hidden()

                if step % self.target_update_steps == 0:

                    self.critic_target.load_state_dict(self.critic.state_dict())

            # Save losses

            actors_losses.append(np.mean(actors_loss))
            critic_losses.append(critic_loss.item())

            # Log the metrics

            t_price_metric, t_emission_metric = self.env.mg.get_houses_metrics()

            train_price_metric.append(t_price_metric.mean())
            train_emission_metric.append(t_emission_metric.mean())

            # Evaluate the model

            e_price_metric, e_emission_metric = self.check_model()

            eval_price_metric.append(e_price_metric.mean())
            eval_emission_metric.append(e_emission_metric.mean())

            # Check stop condition

            stop_condition = np.abs(actors_losses[step]) <= self.min_loss and critic_loss.abs().item() <= self.min_loss
            
            if step % 50 == 0 or stop_condition:

                # Wandb logging

                results = {
                    "rollout_avg_reward": rewards.mean(axis=2).sum(axis=0).mean(),
                    "actors_avg_loss": actor_loss.item(),
                    "critic_avg_loss": critic_loss.item(),
                    "avg_action": actions_hist.mean(),
                }

                self.wdb_logger.log_dict(results)

            # if step % 250 == 0:

            #     # Save networks weights for resume training

            #     self.save_weights(
            #         actor_state_dict=self.actor.state_dict(),
            #         actor_opt_state_dict=self.actor.optimizer.state_dict(),
            #         critic_state_dict=self.critic.state_dict(),
            #         critic_opt_state_dict=self.critic.optimizer.state_dict(),
            #         current_step=step
            #     )

            #     self.wdb_logger.save_model()

        # Return results dictionary

        return {
            "training_steps": self.training_steps,
            "rollout_steps": self.rollout_steps,
            "train": {
                "price_metric": train_price_metric,
                "emission_metric": train_emission_metric,
                "states": all_states,
                "rewards": all_rewards,
                "actions": all_actions,
                "net_energy": all_net_energy
            },
            "eval": {
                "price_metric": eval_price_metric,
                "emission_metric": eval_emission_metric
            },
        }

    def test(self):

        test_price_metric, test_emission_metric = [], []

        for step in tqdm(range(self.current_step, self.training_steps)):

            # Evaluate the model

            e_price_metric, e_emission_metric = self.check_model(mode='test')

            test_price_metric.append(e_price_metric.mean())
            test_emission_metric.append(e_emission_metric.mean())
            
        return {
            "price_metric": test_price_metric,
            "emission_metric": test_emission_metric
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
    model = "d_coma_mg"

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

        plot_metrics(metrics=results)

        # plot_rollout(env=my_env, results=results)
        
        # Finish wandb process

        agent.wdb_logger.finish()

    except (RuntimeError, KeyboardInterrupt):

        traceback.print_exc()