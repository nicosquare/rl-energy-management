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

from src.components.pv import PVParameters, Coordinates, PVCharacteristics
from src.components.load import LoadParameters
from src.components.battery import BatteryParameters
from src.components.generator import GeneratorParameters
from src.components.microgrid import MicrogridArchitecture, MicrogridParameters

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

        state, _, _, _ = self.env.reset()

        for step in range(self.rollout_steps):
            # Start by appending the state to create the states trajectory

            state = state.to(device)
            states.append(state)

            # Perform action and pass to next state

            action, log_prob = self.select_action(Tensor(state))
            state, reward, _, _ = self.env.step(action=action)

            rewards.append(reward)
            log_probs.append(log_prob)

        return states, rewards, log_probs

    def train(self, training_steps: int = 1000):

        for step in range(training_steps):

            # Perform rollouts and sample trajectories

            states, rewards, log_probs = self.rollout()

            log_probs = torch.stack(log_probs, 0)
            value = [self.critic(state) for state in states]

            value = torch.stack(value, 0).squeeze()

            # Causality trick

            sum_rewards = []
            causal_reward = 0

            for reward in reversed(rewards):
                causal_reward = torch.clone(causal_reward + reward)
                sum_rewards.insert(0, causal_reward)

            sum_rewards = torch.stack(sum_rewards, 0)

            # Backpropagation to train Actor NN

            actor_loss = -torch.mean(torch.sum(log_probs * (sum_rewards - value.detach())))
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
            Define the Microgrid parameters
        '''

        exp_pv_params = PVParameters(
            coordinates=Coordinates(
                latitude=24.4274827,
                longitude=54.6234876,
                name='Masdar',
                altitude=0,
                timezone='Asia/Dubai'
            ),
            pv_parameters=PVCharacteristics(
                n_arrays=1,
                modules_per_string=10,
                n_strings=1,
                surface_tilt=20,
                surface_azimuth=180,
                solar_panel_ref='Canadian_Solar_CS5P_220M___2009_',
                inverter_ref='iPower__SHO_5_2__240V_'
            ),
            year=2022
        )
        exp_load_params = LoadParameters(load_type='residential_1')
        exp_battery_params = BatteryParameters(
            soc_0=0.1,
            soc_max=0.9,
            soc_min=0.1,
            p_charge_max=0.5,
            p_discharge_max=0.5,
            efficiency=0.9,
            capacity=4,
            sell_price=0.6,
            buy_price=0.6
        )

        exp_generator_params = GeneratorParameters(
            rated_power=2.5,
            p_max=0.9,
            p_min=0.1,
            fuel_cost=0.4,
            co2=2
        )

        exp_microgrid_arch = MicrogridArchitecture(
            pv=True,
            battery=True,
            generator=True,
            grid=False
        )

        exp_microgrid_params = MicrogridParameters(
            pv=exp_pv_params,
            load=exp_load_params,
            battery=exp_battery_params,
            generator=exp_generator_params,
            grid=None
        )

        '''
            Define the simulation parameters
        '''

        batch_size = 10
        agent_training_steps = 10
        agent_gamma = 0.99
        agent_rollout_steps = 24 * 30  # Hours * Days
        agent_actor_lr = 1e-3
        agent_critic_lr = 1e-3

        '''
            Setup all the configurations for Wandb
        '''

        wandb.init(
            project="cont-a2c-mg-set-gen",
            entity="madog",
            config={
                "batch_size": batch_size,
                "training_steps": agent_training_steps,
                "gamma": agent_gamma,
                "rollout_steps": agent_rollout_steps,
                "agent_actor_lr": agent_actor_lr,
                "agent_critic_lr": agent_critic_lr,
            }
        )

        # Define the custom x-axis metric
        wandb.define_metric("test_step")

        # Define the x-axis for the plots: (avoids an issue with Wandb step autoincrement on each log call)

        wandb.define_metric("test_reward", step_metric='test_step')

        '''
            Run the simulator
        '''

        set_all_seeds(420)

        # Instantiate the environment

        mg_env = MGSetGenerator(mg_arch=exp_microgrid_arch, mg_params=exp_microgrid_params, batch_size=batch_size)

        # Instantiate the agent
        agent = Agent(
            env=mg_env, gamma=agent_gamma, rollout_steps=agent_rollout_steps, critic_lr=agent_actor_lr,
            actor_lr=agent_actor_lr
        )

        # Launch the training

        agent.train(training_steps=agent_training_steps)

        # Test the trained model

        t_state, _, _, _ = agent.env.reset()

        for t_step in range(24 * 365):
            # Perform action and pass to next state

            t_action, _ = agent.select_action(Tensor(t_state))
            t_state, t_reward, _, _ = agent.env.step(action=t_action)

            wandb.log({
                "test_step": t_step,
                "test_action": t_action,
                "test_reward": t_reward
            })

        # Finish wandb process

        wandb.finish()

    except (RuntimeError, KeyboardInterrupt):

        traceback.print_exc()
        wandb.finish()
