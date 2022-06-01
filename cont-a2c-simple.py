"""

    Advantage Actor Critic (A2C) algorithm implementation

    Credits: Nicolás Cuadrado, MBZUAI, OptMLLab

"""
import os
import traceback
import numpy as np
import torch
import wandb

from gym import Env
from torch import Tensor
from torch.nn import Module, Sequential, Linear, LeakyReLU
from torch.optim import Adam
from torch.distributions import Normal
from dotenv import load_dotenv

from src.environments.simple import SimpleEnv

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize Wandb for logging purposes

load_dotenv()
wandb.login(key=str(os.environ.get("WANDB_KEY")))

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


class Critic(Module):

    def __init__(self, num_inputs, hidden_size=64, ):
        super(Critic, self).__init__()

        self.model = Sequential(
            Linear(num_inputs, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, 1)
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.model(state)


class Agent:

    def __init__(
            self, env: Env, gamma: float = 0.99, rollout_steps: int = 5, hidden_size: int = 64,
            actor_lr: float = 1e-4, critic_lr: float = 1e-4,
    ):

        # Parameter initialization

        self.env = env
        self.gamma = gamma
        self.rollout_steps = rollout_steps

        # Configure neural networks

        dim_obs = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0]

        self.actor = Actor(num_inputs=dim_obs, num_actions=dim_action, hidden_size=hidden_size).to(device)
        self.critic = Critic(num_inputs=dim_obs, hidden_size=hidden_size).to(device)

        self.actor.optimizer = Adam(params=self.actor.parameters(), lr=actor_lr)
        self.critic.optimizer = Adam(params=self.critic.parameters(), lr=critic_lr)

        # Hooks into the models to collect gradients and topology

        wandb.watch(models=(self.actor, self.critic))

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

            states, rewards, log_probs = self.rollout()

            sum_rewards = torch.sum(torch.stack(rewards, dim=0), dim=0)
            sum_log_probs = torch.sum(torch.stack(log_probs, dim=0), dim=0)

            # Just considering the estimated Q value according to the initial state on each trajectory

            value = self.critic(states[0])

            # Backpropagation to train Actor NN

            actor_loss = -torch.mean(sum_log_probs*(sum_rewards - value.detach().T))
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Backpropagation to train Critic NN

            critic_loss = torch.mean((value - sum_rewards) ** 2)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            wandb.log({
                "rollout_avg_reward": torch.mean(sum_rewards),
                "actor_loss": actor_loss,
                "critic_loss": critic_loss
            })


if __name__ == '__main__':

    try:
        
        '''
            Define the simulation parameters
        '''

        batch_size = 64
        agent_training_steps = 500
        agent_gamma = 0.99
        agent_actor_lr = 1e-3
        agent_critic_lr = 3e-3

        '''
            Setup all the configurations for Wandb
        '''

        wandb.init(
            project="cont-a2c-simple",
            entity="madog",
            config={
                "batch_size": batch_size,
                "training_steps": agent_training_steps,
                "gamma": agent_gamma,
                "agent_actor_lr": agent_actor_lr,
                "agent_critic_lr": agent_critic_lr,
            }
        )
        
        '''
            Run the simulator
        '''

        set_all_seeds(420)

        # Instantiate the environment

        my_env = SimpleEnv(batch_size=batch_size)
        
        # Instantiate the agent
        agent = Agent(
            env=my_env, gamma=agent_gamma, critic_lr=agent_critic_lr, actor_lr=agent_actor_lr
        )

        # Launch the training

        agent.train(training_steps=agent_training_steps)

        # Finish wandb process

        wandb.finish()

    except (RuntimeError, KeyboardInterrupt):

        traceback.print_exc()
        wandb.finish()