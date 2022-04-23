"""

    Advantage Actor Critic (A2C) algorithm implementation

    Credits: Nicolás Cuadrado, MBZUAI, OptMLLab

"""
import os
import numpy as np
import torch
import wandb

from gym import Env
from torch import Tensor
from torch.nn import Module, Sequential, Linear, LeakyReLU
from torch.optim import Adam
from torch.distributions import Normal
from dotenv import load_dotenv

from src.environments.mg_set_generator import MGSetGenerator

torch.set_default_dtype(torch.float64)

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

    def forward(self, state: Tensor) -> (Tensor, Tensor):
        mu, sigma = self.model(state)

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
        self.current_state = None

        # Configure neural networks

        dim_obs = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0]

        self.actor = Actor(num_inputs=dim_obs, num_actions=dim_action, hidden_size=hidden_size)
        self.critic = Critic(num_inputs=dim_obs, hidden_size=hidden_size)

        self.actor.optimizer = Adam(params=self.actor.parameters(), lr=actor_lr)
        self.critic.optimizer = Adam(params=self.critic.parameters(), lr=critic_lr)

        # Set initial conditions

        self.state, _, _, _ = self.env.reset()

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

        # Perform the rollouts

        for step in range(self.rollout_steps):
            states.append(self.state)

            # Perform action and pass to next state

            action, log_prob = self.select_action(Tensor(self.state))
            self.state, reward, _, _ = self.env.step(action=action)

            wandb.log({
                "reward": reward,
                "action": action
            })

            rewards.append(reward)
            log_probs.append(log_prob)

        return states, rewards, log_probs

    def train(self, training_steps: int = 1000):

        for step in range(training_steps):
            # Perform rollouts and sample trajectories

            states, rewards, log_probs = self.rollout()

            sum_rewards = np.sum(rewards)
            sum_log_probs = torch.sum(torch.cat([log_prob.unsqueeze(0) for log_prob in log_probs]))

            value = self.critic(Tensor(states[0]))

            # Backpropagation to train Actor NN

            actor_loss = - sum_log_probs * (sum_rewards - value.detach())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Backpropagation to train Critic NN

            critic_loss = (value - sum_rewards) ** 2
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            wandb.log({
                "actor_loss": actor_loss,
                "critic_loss": critic_loss
            })


if __name__ == '__main__':

    try:
        '''
            Define the simulation parameters
        '''

        agent_training_steps = 1000
        agent_gamma = 0.99
        agent_rollout_steps = 24 * 30 * 12  # Hours * Days * Months
        agent_actor_lr = 1e-4
        agent_critic_lr = 1e-4

        '''
            Setup all the configurations for Wandb
        '''

        wandb.init(
            project="cont-a2c-mg-set-gen",
            entity="madog",
            config={
                "num_frames": agent_training_steps,
                "gamma": agent_gamma,
                "n_steps": agent_rollout_steps,
                "agent_actor_lr": agent_actor_lr,
                "agent_critic_lr": agent_critic_lr,
            }
        )

        # Define the custom x-axis metric
        wandb.define_metric("current_t")

        # Define the x-axis for the plots: (avoids an issue with Wandb step autoincrement on each log call)

        wandb.define_metric("load", step_metric='current_t')
        wandb.define_metric("pv", step_metric='current_t')
        wandb.define_metric("generator", step_metric='current_t')
        wandb.define_metric("remaining_power", step_metric='current_t')
        wandb.define_metric("unattended_power", step_metric='current_t')
        wandb.define_metric("soc", step_metric='current_t')
        wandb.define_metric("cap_to_charge", step_metric='current_t')
        wandb.define_metric("cap_to_discharge", step_metric='current_t')
        wandb.define_metric("p_charge", step_metric='current_t')
        wandb.define_metric("p_discharge", step_metric='current_t')
        wandb.define_metric("load", step_metric='current_t')
        wandb.define_metric("cost", step_metric='current_t')
        wandb.define_metric("reward", step_metric='current_t')
        wandb.define_metric("action", step_metric='current_t')
        wandb.define_metric("actor_loss", step_metric='current_t')
        wandb.define_metric("critic_loss", step_metric='current_t')

        '''
            Run the simulator
        '''

        set_all_seeds(420)

        # Instantiate the environment

        mg_env = MGSetGenerator()

        # Instantiate the agent
        agent = Agent(
            env=mg_env, gamma=agent_gamma, rollout_steps=agent_rollout_steps, critic_lr=agent_actor_lr,
            actor_lr=agent_actor_lr
        )

        # Launch the training

        agent.train(training_steps=agent_training_steps)

    except KeyboardInterrupt:
        wandb.finish()
