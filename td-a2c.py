"""

    TD-Lambda Actor Critic algorithm

    Inspired by the implementation in Stable-Baselines 3, our own source code to apply this technique.

    Credits: Rubén Soloazabal, MBZUAI, Optim-Lab

"""
import os
import numpy as np
import torch
import wandb
import torch.nn.functional as func
import torch.optim as optim

from typing import Tuple
from math import floor
from torch import clamp, Tensor
from torch.nn import ReLU, Linear, Module, Sequential, Sigmoid
from torch.distributions import Normal
from dotenv import load_dotenv

from src.environments.mg_set_generator import MGSetGenerator

torch.set_default_dtype(torch.float64)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize Wandb for logging purposes

load_dotenv()
wandb.login(key=str(os.environ.get("WANDB_KEY")))


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
            ReLU(),
            Linear(hidden_size, hidden_size),
            Sigmoid(),
            Linear(hidden_size, num_actions)  # Mu layer
        )

    def forward(self, state: Tensor) -> Normal:
        mean, stds = self.model(state)

        mean = clamp(mean, 1e-3, 1)
        stds = clamp(stds, 1e-3, 1)
        dist = Normal(mean, stds)

        return dist


class Critic(Module):
    def __init__(self, num_inputs, hidden_size=64):
        super(Critic, self).__init__()

        self.model = Sequential(
            Linear(num_inputs, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, 1)
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.model(state)


class A2CAgent:

    def __init__(self, env, gamma: float, entropy_weight: float, n_steps: int, hidden_size: int = 64, lr: float = 1e-4):
        """Initialize."""
        self.env = env
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.n_steps = n_steps

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.actor = Actor(obs_dim, action_dim, hidden_size).to(device)
        self.critic = Critic(obs_dim, hidden_size).to(device)

        # Hooks into the models to collect gradients and topology

        wandb.watch(models=(self.actor, self.critic))

        self.state = None

        # optimizer
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # transition (state, log_prob, next_state, reward, done)
        self.transitions: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def env_reset(self):

        state, _, _, _ = self.env.reset()
        self.state = torch.tensor(state, device=device)

        return state

    def env_restore(self, time_step: int):
        self.total_step = time_step
        self.env.restore(time_step=time_step)

    def select_action(self, state):
        """Select an action from the input state."""

        dist = self.actor(state)

        # Compute the action
        m = dist
        action = m.sample()
        entropy = m.entropy().cpu().detach().numpy()

        if not self.is_test:
            log_prob = m.log_prob(action)
            self.transitions.append([state, log_prob])

        return action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transitions[-1].extend([next_state, reward, done])

        return next_state, reward, done

    def rollout(self):
        """Compute the n-step rollout."""

        # Reset transitions
        self.transitions: list = list()

        for step in range(self.n_steps):

            action = self.select_action(self.state)
            value = self.critic(self.state)

            if not self.is_test:
                self.transitions[-1].extend(value)

            next_state, reward, done = agent.step(action)

            wandb.log({
                "reward": reward,
                "action": action
            })

            self.state = torch.tensor(next_state, device=device)
            if done:
                self.env_reset()

        # Update number of steps
        agent.total_step += self.n_steps

        return agent.transitions

    def returns_and_advantages(self, transitions):
        """Returns and advantages."""

        states, log_probs, values, next_states, rewards, dones = list(zip(*transitions))

        returns = np.zeros(shape=self.n_steps)
        advantages = np.zeros(shape=self.n_steps)

        with torch.no_grad():
            # Compute value for the last timestep
            next_state = next_states[-1]
            next_state = torch.tensor(next_state, device=device)
            last_value = self.critic(next_state)

        tmp = 0
        for step in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - dones[step]
            if step == self.n_steps - 1:
                next_values = last_value.detach()
            else:
                next_values = values[step + 1]

            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step].detach()

            tmp = delta + self.gamma * next_non_terminal * tmp
            advantages[step] = tmp
            returns[step] = advantages[step] + values[step].detach()

        return returns, advantages


if __name__ == '__main__':

    try:

        '''
            Define the simulation parameters
        '''

        num_frames = 24 * 30 * 12  # Hours * Days * Months
        agent_gamma = 0.99
        agent_entropy_weight = 1e-2
        agent_n_steps = 5
        agent_learning_rate = 4e-4
        num_updates_to_test = 100
        test_batch = 5
        num_frames_per_trajectory = 24 * 30 * 1  # Hours * Days * Months

        '''
            Setup all the configurations for Wandb
        '''

        wandb.init(
            project="td-a2c-mg-set-gen",
            entity="madog",
            config={
                "num_frames": num_frames,
                "gamma": agent_gamma,
                "entropy_weight": agent_entropy_weight,
                "n_steps": agent_n_steps,
                "learning_rate": agent_learning_rate,
                "batch": test_batch,
                "num_updates_to_test": num_updates_to_test,
                "num_frames_per_trajectory": num_frames_per_trajectory,
            }
        )

        # Define the custom x-axis metric
        wandb.define_metric("current_t")

        # Define the x-axis for the plots: (avoids an issue with Wandb step autoincrement on each log call)

        wandb.define_metric("load", step_metric='current_t')
        wandb.define_metric("pv", step_metric='current_t')
        wandb.define_metric("generator", step_metric='current_t')
        wandb.define_metric("remaining_power", step_metric='current_t')
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
        wandb.define_metric("loss", step_metric='current_t')
        wandb.define_metric("test_batch_reward", step_metric='current_t')

        '''
            Run the simulator
        '''

        set_all_seeds(420)

        # Instantiate the environment

        mg_env = MGSetGenerator()

        # Instantiate the agent
        agent = A2CAgent(
            env=mg_env, gamma=agent_gamma, entropy_weight=agent_entropy_weight, n_steps=agent_n_steps, lr=agent_learning_rate
        )

        """Train the agent"""
        agent.is_test = False

        actor_losses, critic_losses, scores = [], [], []
        agent.env_reset()
        policy_updates = 0

        while agent.total_step < num_frames:

            agent_transitions = agent.rollout()
            agent_states, agent_log_probs, agent_values, agent_next_states, agent_rewards, agent_dones = list(
                zip(*agent_transitions)
            )

            agent_returns, agent_advantages = agent.returns_and_advantages(agent_transitions)

            value_loss = func.mse_loss(torch.stack(agent_values), torch.tensor(agent_returns, device=device))

            policy_loss = -(torch.tensor(agent_advantages, device=device) * torch.stack(agent_log_probs)).mean()

            loss = policy_loss + 0.5 * value_loss

            # update policy
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

            actor_loss, critic_loss = policy_loss.item(), value_loss.item()

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            wandb.log({
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "loss": loss
            })

            policy_updates += 1

            # Test current policy with a batch of trajectories

            if policy_updates % num_updates_to_test == 0:
                sc = []

                # Save current time step to continue where it was after testing

                training_current_t = agent.total_step
                agent.env.set_logging(enabled=False)

                for i in range(test_batch):

                    agent_done = False
                    agent_state = agent.env_reset()
                    agent_rewards = []

                    # Test with trajectories of a year. If less than a year of num_frames size

                    for _ in range(num_frames_per_trajectory):

                        agent_state = torch.tensor(agent_state).to(device)
                        agent_action = agent.select_action(agent_state)
                        agen_next_state, agent_reward, agent_done = agent.step(agent_action)
                        agent_state = agen_next_state
                        agent_rewards.append(agent_reward)

                    sc.append(sum(agent_rewards))

                scores.append(sum(sc) / test_batch)

                agent.env_reset()

                wandb.log({
                    "test_batch_reward": scores[-1],
                })

                # Restore the agent current time step to continue with the training

                agent.env_restore(training_current_t)
                agent.env.set_logging(enabled=True)

        # Finish Wandb process when finishing

        wandb.finish()

    except KeyboardInterrupt:
        wandb.finish()
