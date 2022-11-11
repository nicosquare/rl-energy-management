"""

    Policy Gradient with causality algorithm implementation

    Credits: Nicolás Cuadrado, Alejandro Gutierrez, MBZUAI, OptMLLab

"""

import traceback
import numpy as np
import torch
import argparse
from tqdm import tqdm

from gym import Env
from torch import Tensor, tensor
from torch.nn import Module, Linear
from torch.functional import F
from torch.optim import Adam
from torch.distributions import Categorical

from src.utils.wandb_logger import WandbLogger
from src.environments.mg_simple import MGSimple
from src.utils.tools import set_all_seeds, load_config

torch.autograd.set_detect_anomaly(True)

# Define global variables

ZERO = 1e-5

'''
    Agent definitions
'''
class Actor(Module):

    def __init__(self, state_size, num_actions, hidden_size=64):
        super(Actor, self).__init__()

        # Define the independent inputs

        self.input = Linear(state_size, hidden_size)
        self.output = Linear(hidden_size, num_actions)

    def forward(self, state: Tensor) -> Tensor:

        state = F.relu(self.input(state))

        output = F.softmax(self.output(state), dim=1)

        return output

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
        self.random_soc_0 = config['env']['random_soc_0']
        self.central_agent = config['env']['central_agent']
        self.disable_noise = config['env']['disable_noise']
        
        config = config['agent']
        # Get params from yaml config file
        self.num_disc_act = config['num_disc_act']
        self.actor_lr = config['actor_lr']
        self.actor_nn = config['actor_nn']
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
            "agent_actor_nn": self.actor_nn,
            "gamma": self.gamma,
            "central_agent": self.central_agent,
            "random_soc_0": self.random_soc_0,
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
        else:
            self.device = torch.device("cpu")
            print("Running on CPU")

        # Configure predictors

        if self.extended_observation:

            print('This model does not support extended observations yet')

            num_inputs = env.obs_size

        else:

            num_inputs = env.obs_size

        # Configure neural networks

        self.actor = Actor(state_size=num_inputs, num_actions=len(self.discrete_actions), hidden_size=self.actor_nn).to(self.device)
        self.actor.optimizer = Adam(params=self.actor.parameters(), lr=self.actor_lr)

        # Check if we are resuming training from a previous checkpoint

        if resumed:

            checkpoint = torch.load(self.wdb_logger.load_model().name)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor.optimizer.load_state_dict(checkpoint['actor_opt_state_dict'])
            self.current_step = checkpoint['current_step']

        self.actor.train()

        # Hooks into the models to collect gradients and topology

        self.wdb_logger.watch_model(models=(self.actor))

    def setup_wandb_logger(self, config: dict, tags: list):
        
        wdb_logger = WandbLogger(tags=tags)

        wdb_logger.disable_logging(self.disable_logging)

        wdb_logger.init(config=config)

        return wdb_logger

    def select_action(self, state: Tensor):

        probs = self.actor(state)

        # Define the distribution

        dist = Categorical(probs=probs)

        # Sample action
        
        action_index = dist.sample()
        action = np.stack([[self.discrete_actions[batch_index.cpu()]] for batch_index in action_index])

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

    def train(self):

        all_states, all_rewards, all_actions, all_net_energy = [], [], [], []

        for step in tqdm(range(self.current_step, self.training_steps)):

            # Perform rollouts and sample trajectories

            states, rewards, log_probs, actions_hist = self.rollout()

            # Append the trajectories to the arrays

            all_states.append(states)
            all_rewards.append(rewards)
            all_actions.append(actions_hist)
            all_net_energy.append(self.env.mg.net_energy)

            # Process the current trajectory

            sum_log_probs = torch.stack(log_probs, dim=0).sum(dim=0)
            sum_rewards = tensor(np.sum(rewards, axis=0).squeeze(axis=-1)).to(self.device)

            # Perform the optimization step

            try:

                actor_loss = - torch.mean(sum_log_probs*sum_rewards, dim=0)

                # Backpropagation to train Actor NN

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            except Exception as e:

                traceback.print_exc()

            # Check stop condition

            stop_condition = actor_loss.abs().item() <= self.min_loss

            if step % 50 == 0 or stop_condition:

                # Wandb logging

                results = {
                    "rollout_avg_reward": rewards.mean(axis=1).sum(axis=0)[0],
                    "actor_loss": actor_loss.item(),
                    "avg_action": actions_hist.mean(),
                }

                self.wdb_logger.log_dict(results)

            if step % 250 == 0:

                # Save networks weights for resume training

                self.save_weights(
                    actor_state_dict=self.actor.state_dict(),
                    actor_opt_state_dict=self.actor.optimizer.state_dict(),
                    current_step=step
                )

                self.wdb_logger.save_model()

        return all_states, all_rewards, all_actions, all_net_energy

    # Save weights to file

    def save_weights(self, actor_state_dict, actor_opt_state_dict, current_step):

        model_path = self.wdb_logger.run.dir if self.wdb_logger.run is not None else './models'

        torch.save({
            'current_step': current_step,
            'actor_state_dict': actor_state_dict,
            'actor_opt_state_dict': actor_opt_state_dict,
        }, f'{model_path}/d_pg_model.pt')

        print(f'Saving model on step: {current_step}')


"""
    Main method definition
"""

if __name__ == '__main__':
    config = load_config("d_pg")
    config = config['train']
    
    # Start wandb logger

    try:
        '''
            Run the simulator
        '''

        set_all_seeds(0)

        # Instantiate the environment

        my_env = MGSimple(config=config['env'])

        # Instantiate the agent

        agent = Agent(
            env=my_env, config = config
        )

        # Launch the training

        agent.train()

        # Finish wandb process

        agent.wdb_logger.finish()

    except (RuntimeError, KeyboardInterrupt):

        traceback.print_exc()