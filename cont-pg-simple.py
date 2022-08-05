"""

    Policy Gradient algorithm implementation

    Credits: Nicolás Cuadrado, MBZUAI, OptMLLab

"""

import traceback
import numpy as np
import torch
import argparse

from gym import Env
from torch import Tensor
from torch.nn import Module, Sequential, Linear, LeakyReLU
from torch.optim import Adam
from torch.distributions import Normal

from src.environments.simple import SimpleEnv
from src.utils.wandb_logger import WandbLogger

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

wdb_logger = WandbLogger(project_name="cont-pg-simple", entity_name="madog")

# Define global variables

zero = 1e-5


# Misc. methods

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Actor(Module):

    def __init__(self, num_inputs, num_actions, hidden_size=64):
        super(Actor, self).__init__()

        self.model = Sequential(
            Linear(num_inputs, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, num_actions * 2)  # For each continuous action, a mu and a sigma
        )

    def forward(self, state: Tensor):
        normal_params = self.model(state)

        mu = normal_params[:, 0]
        sigma = normal_params[:, 1]

        # Guarantee that the standard deviation is not negative

        sigma = torch.exp(sigma) + zero

        return mu, sigma

class Agent:

    def __init__(
            self, env: Env, gamma: float = 0.99, rollout_steps: int = 5, hidden_size: int = 64,
            actor_lr: float = 1e-4,
    ):

        # Parameter initialization

        self.env = env
        self.gamma = gamma
        self.rollout_steps = rollout_steps

        # Configure neural networks

        dim_obs = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0]

        self.actor = Actor(num_inputs=dim_obs, num_actions=dim_action, hidden_size=hidden_size).to(device)

        self.actor.optimizer = Adam(params=self.actor.parameters(), lr=actor_lr)

        # Hooks into the models to collect gradients and topology

        wdb_logger.watch_model(models=(self.actor))

    def select_action(self, state: Tensor):

        mu, sigma = self.actor(state)

        dist = Normal(loc=mu, scale=sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def rollout(self):

        states, rewards, log_probs = [], [], []

        # Get the initial state by resetting the environment

        state, _, _, done = self.env.reset()

        while not done:
            
            # Start by appending the state to create the states trajectory

            state = Tensor(state).to(device)
            states.append(state)

            # Perform action and pass to next state

            action, log_prob = self.select_action(state)
            state, reward, _, done = self.env.step(action=action)

            rewards.append(Tensor(reward))
            log_probs.append(log_prob)

        return states, rewards, log_probs

    def train(self, training_steps: int = 1000):

        for _ in range(training_steps):

            # Perform rollouts and sample trajectories

            _, rewards, log_probs = self.rollout()

            sum_rewards = torch.sum(torch.stack(rewards, dim=0), dim=0).to(device)
            sum_log_probs = torch.sum(torch.stack(log_probs, dim=0), dim=0)

            # Backpropagation to train Actor NN

            actor_loss = -torch.mean(sum_log_probs*sum_rewards)
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            wdb_logger.log_dict({
                "rollout_avg_reward": torch.mean(sum_rewards),
                "actor_loss": actor_loss
            })


"""
    Main method definition
"""

# Read arguments from command line

parser = argparse.ArgumentParser(prog='rl', description='RL Experiments')

args = parser.parse_args([])

parser.add_argument("-dl", "--disable_logging", default=False, action="store_false", help="Disable logging")
parser.add_argument("-b", "--batch_size", default=64, type=int, help="Batch size (ideally a multiple of 2)")
parser.add_argument("-ts", "--training_steps", default=500, type=int, help="Steps for training loop")
parser.add_argument("-g", "--gamma", default=0.99, type=float, help="Reward discount factor")
parser.add_argument("-alr", "--actor_lr", default=1e-3, type=float, help="Actor learning rate")

args = parser.parse_args()

if __name__ == '__main__':

    # Get arguments from command line

    disable_logging = args.disable_logging
    training_steps = args.training_steps
    batch_size = args.batch_size
    gamma = args.gamma
    actor_lr = args.actor_lr

    try:
        
        '''
            Define the simulation parameters
        '''

        agent_batch_size = batch_size
        agent_training_steps = training_steps
        agent_gamma = gamma
        agent_actor_lr = actor_lr

        '''
            Setup all the configurations for Wandb
        '''

        wdb_logger.disable_logging(disable=disable_logging)

        wdb_logger.init(config={
            "batch_size": agent_batch_size,
            "training_steps": agent_training_steps,
            "gamma": agent_gamma,
            "agent_actor_lr": agent_actor_lr,
        })
        
        '''
            Run the simulator
        '''

        set_all_seeds(420)

        # Instantiate the environment

        my_env = SimpleEnv(batch_size=agent_batch_size)
        
        # Instantiate the agent
        agent = Agent(
            env=my_env, gamma=agent_gamma, actor_lr=agent_actor_lr
        )

        # Launch the training

        agent.train(training_steps=agent_training_steps)

    except (RuntimeError, KeyboardInterrupt):

        traceback.print_exc()

    finally:

        # Finish wandb process

        wdb_logger.finish()