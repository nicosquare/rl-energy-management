"""

    Advantage Actor Critic (A2C) with causality algorithm implementation

    Credits: Nicolás Cuadrado, Alejandro Gutierrez, MBZUAI, OptMLLab

"""

import traceback
import numpy as np
import torch
import argparse
from tqdm import tqdm

from gym import Env
from torch import Tensor, tanh
from torch.nn import Module, Sequential, Linear, LeakyReLU, MSELoss
from torch.optim import Adam
from torch.distributions import Normal
from citylearn.preprocessing import OnehotEncoding

from src.utils.wandb_logger import WandbLogger
from src.environments.simple import SimpleEnv

torch.autograd.set_detect_anomaly(True)

wdb_logger = WandbLogger(tags=["a2c-caus","batches", "mean_reward", "simple"])

# Define global variables

zero = 1e-5

'''
    Agent definitions
'''

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

        mu = normal_params[:, :, 0]
        sigma = normal_params[:, :, 1]

        # Guarantee that the standard deviation is not negative

        sigma = torch.exp(sigma) + zero

        return mu, sigma


class Critic(Module):

    def __init__(self, num_inputs, hidden_size=64):
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
        self, env: Env, actor_nn: int = 64, critic_nn: int = 64, actor_lr: float = 1e-4, critic_lr: float = 1e-4, gamma: float = 0.9,
        batch_size: int = 1, resumed: bool=False, extended_obs: bool=False
    ):

        # Parameter initialization

        self.env = env
        self.batch_size = batch_size
        self.resumed = resumed
        self.current_step = 0
        self.gamma = gamma
        self.extended_obs = extended_obs
        
        if extended_obs:
            num_inputs = env.dim_obs + self.env.dim_action
        else:
            num_inputs = env.dim_obs

        # Configure neural networks
        
        self.actor = Actor(num_inputs=num_inputs, num_actions=env.dim_action, hidden_size=actor_nn).to(device)
        self.critic = Critic(num_inputs=num_inputs, hidden_size=critic_nn).to(device)

        self.actor.optimizer = Adam(params=self.actor.parameters(), lr=actor_lr)
        self.critic.optimizer = Adam(params=self.critic.parameters(), lr=critic_lr)

        # Check if we are resuming training from a previous checkpoint

        if resumed:

            checkpoint = torch.load(wdb_logger.load_model().name)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor.optimizer.load_state_dict(checkpoint['actor_opt_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic.optimizer.load_state_dict(checkpoint['critic_opt_state_dict'])
            self.current_step = checkpoint['current_step']
            
        self.actor.train()
        self.critic.train()

        # Hooks into the models to collect gradients and topology

        wdb_logger.watch_model(models=(self.actor, self.critic))

    def select_action(self, state: Tensor):

        mu, sigma = self.actor(state)

        # Define the distribution

        dist = Normal(loc=mu, scale=sigma)

        # Transform the distribution to restrict the range of the output

        action = dist.sample()

        log_prob = dist.log_prob(action).squeeze() # Squeeze for the case of central agent

        # Reshape to fullfill env requirements

        action = action.reshape(self.batch_size, self.env.dim_action, 1).cpu().numpy()

        return action, log_prob

    def get_extended_observations(self, state):

        encoder = OnehotEncoding(classes=range(self.env.dim_action))

        state_extension = np.ones((state.shape[0],state.shape[1], 1)) * [encoder*i for i in range(self.env.dim_action)]

        # Build extended state

        return np.concatenate((state, state_extension), axis=2)

    def rollout(self):

        states, rewards, log_probs, actions_hist = [], [], [], []

        # Get the initial state by resetting the environment

        state, _, _, done = self.env.reset()

        if self.extended_obs:

            state = self.get_extended_observations(state)

        # Launch rollout

        while not done:
            
            # Start by appending the state to create the states trajectory

            state = Tensor(state).to(device)
            states.append(state)

            # Perform action and pass to next state

            actions, log_prob = self.select_action(state)

            state, reward, _, done = self.env.step(actions)

            if self.extended_obs:

                state = self.get_extended_observations(state)

            rewards.append(reward)
            log_probs.append(log_prob)
            actions_hist.append(actions)

        rewards = np.stack(rewards)
        actions_hist = np.vstack(actions_hist)

        return states, rewards, log_probs, actions_hist

    def train(self, training_steps: int = 1000, epsilon: float = 0.5):

        # Arrays to follow the actions form the agent for each house

        for step in tqdm(range(self.current_step, training_steps)):

            # Perform rollouts and sample trajectories

            states, rewards, log_probs, actions_hist = self.rollout()

            # Perform the optimization step

            log_probs = torch.stack(log_probs, dim=0)

            states = torch.stack(states, dim=0)
            value = self.critic(states).squeeze(dim=3)

            # Causality trick considering gamma

            sum_rewards = []
            prev_reward = 0

            for reward in reversed(rewards):
                prev_reward = np.copy(reward + self.gamma * prev_reward)
                sum_rewards.insert(0, prev_reward+0.0)

            sum_rewards = Tensor(np.stack(sum_rewards)).to(device)

            # Perform the optimization step

            try:

                # Backpropagation to train Actor NN

                actor_loss = -torch.mean(torch.sum(log_probs*(sum_rewards - value.detach()), dim=0))
                actor_loss.backward()
                self.actor.optimizer.step()
                self.actor.optimizer.zero_grad()

                # Backpropagation to train Critic NN

                critic_loss = MSELoss()(value, sum_rewards)
                critic_loss.backward()
                self.critic.optimizer.step()
                self.critic.optimizer.zero_grad()

            except Exception as e:

                traceback.print_exc()

            # Check stop condition

            stop_condition = actor_loss.abs().item() <= epsilon and critic_loss.abs().item() <= epsilon

            if step % 5 == 0 or stop_condition:

                # Wandb logging

                results = {
                    "rollout_avg_reward": torch.mean(sum_rewards).item(),
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item()
                }

                # Add each house action to the results

                actions_hist = actions_hist.mean(axis=2) # Average over the batch
                actions_hist_mean = actions_hist.mean(axis=0)

                for i in range(actions_hist.shape[1]):
                    results[f"house_{i}_avg_action"] = actions_hist_mean[i]
                
                wdb_logger.log_dict(results)

            if step % 50 == 0 or stop_condition:

                # Save networks weights for resume training

                self.save_weights(
                    actor_state_dict=self.actor.state_dict(),
                    actor_opt_state_dict=self.actor.optimizer.state_dict(),
                    critic_state_dict=self.critic.state_dict(),
                    critic_opt_state_dict=self.critic.optimizer.state_dict(),
                    current_step=step
                )

                wdb_logger.save_model()

                if stop_condition:

                    print(f"Converged at step {step}. Exiting...")
                    break

    # Save weights to file

    def save_weights(self, actor_state_dict, actor_opt_state_dict, critic_state_dict, critic_opt_state_dict, current_step):

        model_path = wdb_logger.run.dir if wdb_logger.run is not None else './models'

        torch.save({
            'current_step': current_step,
            'actor_state_dict': actor_state_dict,
            'actor_opt_state_dict': actor_opt_state_dict,
            'critic_state_dict': critic_state_dict,
            'critic_opt_state_dict': critic_opt_state_dict,
        }, f'{model_path}/model.pt')

        print(f'Saving model on step: {current_step}')

"""
    Main method definition
"""

# Read arguments from command line

parser = argparse.ArgumentParser(prog='rl', description='RL Experiments')

args = parser.parse_args([])

parser.add_argument("-dl", "--disable_logging", default=False, action="store_true", help="Disable logging")
parser.add_argument("-bs", "--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("-ts", "--training_steps", default=500, type=int, help="Steps for training loop")
parser.add_argument("-alr", "--actor_lr", default=1e-3, type=float, help="Actor learning rate")
parser.add_argument("-clr", "--critic_lr", default=1e-3, type=float, help="Critic learning rate")
parser.add_argument("-ann", "--actor_nn", default=256, type=int, help="Actor hidden layer number of neurons")
parser.add_argument("-cnn", "--critic_nn", default=256, type=int, help="Critic hidden layer number of neurons")
parser.add_argument("-g", "--gamma", default=0.95, type=float, help="Critic hidden layer number of neurons")
parser.add_argument("-gpu", "--enable_gpu", default=False, action="store_true", help="Device to use for training")
parser.add_argument("-xobs", "--extended_observation", default=False, action="store_true", help="Extended observation")
parser.add_argument("-eps", "--epsilon", default=0.5, type=float, help="Epsilon for stop condition")

args = parser.parse_args()

if __name__ == '__main__':

    # Get arguments from command line

    disable_logging = args.disable_logging
    batch_size = args.batch_size
    training_steps = args.training_steps
    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    actor_nn = args.actor_nn
    critic_nn = args.critic_nn
    gamma = args.gamma
    enable_gpu = args.enable_gpu
    extended_observation = args.extended_observation
    epsilon = args.epsilon
    
    # Enable GPU if available

    if enable_gpu and torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Running on GPU")
    else:
            device = torch.device("cpu")
            print("Running on CPU")

    # Start wandb logger

    try:

        '''
            Setup all the configurations for Wandb
        '''

        wdb_logger.disable_logging(disable=disable_logging)

        wdb_logger.init(config={
            "training_steps": training_steps,
            "batch_size": batch_size,
            "agent_actor_lr": actor_lr,
            "agent_critic_lr": critic_lr,
            "agent_actor_nn": actor_nn,
            "agent_critic_nn": critic_nn,
            "gamma": gamma,
            "extended_observation": extended_observation,
            "epsilon": epsilon
        })

        '''
            Run the simulator
        '''

        set_all_seeds(0)

        # Instantiate the environment

        my_env = SimpleEnv(
            batch_size=batch_size, n_actions=5
        )
        
        # Instantiate the agent

        agent = Agent(
            env=my_env, critic_lr=critic_lr, actor_lr=actor_lr, actor_nn=actor_nn, critic_nn=critic_nn, batch_size = batch_size, gamma=gamma,
            extended_obs=extended_observation
        )

        # Launch the training

        agent.train(training_steps=training_steps, epsilon=epsilon)

    except (RuntimeError, KeyboardInterrupt):

        traceback.print_exc()
    
    finally:

        # Finish wandb process

        wdb_logger.finish()