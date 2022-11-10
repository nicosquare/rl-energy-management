"""

    Advantage Actor Critic (A2C) with causality algorithm implementation

    Credits: Nicolás Cuadrado, Alejandro Gutierrez, MBZUAI, OptMLLab

"""

from os import path
import traceback
import yaml
import numpy as np
import torch
import argparse
from tqdm import tqdm

from gym import Env
from torch import Tensor, tensor
from torch.nn import Module, Linear, MSELoss
from torch.functional import F
from torch.optim import Adam
from torch.distributions import Categorical

from src.utils.wandb_logger import WandbLogger
from src.environments.mg_simple import MGSimple

torch.autograd.set_detect_anomaly(True)

# Define global variables

CONFIG_PATH = "config/"
ZERO = 1e-5

'''
    Agent definitions
'''

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Function to load yaml configuration file

def load_config(config_name):
    with open(path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

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

# class Actor(Module):

#     def __init__(self, state_size, num_actions, hidden_size=64):
#         super(Actor, self).__init__()

#         # Define the independent inputs

#         self.input = Linear(state_size, hidden_size)
#         self.fc_1 = Linear(hidden_size, hidden_size * 2)
#         self.output = Linear(hidden_size * 2, num_actions)

#     def forward(self, state: Tensor) -> Tensor:

#         state = F.relu(self.input(state))
#         state = F.relu(self.fc_1(state))

#         output = F.softmax(self.output(state), dim=1)

#         return output


class Critic(Module):

    def __init__(self, state_size, hidden_size=64):
        super(Critic, self).__init__()

        # Define the independent inputs

        self.input = Linear(state_size, hidden_size)
        self.output = Linear(hidden_size, 1)

    def forward(self, state: Tensor) -> Tensor:

        state = F.leaky_relu(self.input(state))

        output = self.output(state)

        return output

class Agent:

    def __init__(
        self, env: Env, actor_nn: int = 64, critic_nn: int = 64, actor_lr: float = 1e-4, critic_lr: float = 1e-4, gamma: float = 0.9,
        batch_size: int = 1, resumed: bool = False, extended_obs: bool = False, disable_wandb: bool = False, wandb_dict: dict = None,
        enable_gpu: bool = False, num_disc_act: int = 40,
    ):

        self.discrete_actions = np.linspace(-0.9, 0.9, num_disc_act)
        
        # Parameter initialization

        self.env = env
        self.batch_size = batch_size
        self.resumed = resumed
        self.current_step = 0
        self.gamma = gamma
        self.extended_obs = extended_obs

        self.wdb_logger = self.setup_wandb_logger(config=wandb_dict, tags=["a2c-caus", "discrete"], disabled=disable_wandb)

        # Enable GPU if available

        if enable_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Running on GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on CPU")

        # Configure predictors

        if extended_obs:

            print('This model does not support extended observations yet')

            num_inputs = env.obs_size

        else:

            num_inputs = env.obs_size

        # Configure neural networks

        self.actor = Actor(state_size=num_inputs, num_actions=len(self.discrete_actions), hidden_size=actor_nn).to(self.device)
        self.critic = Critic(state_size=num_inputs, hidden_size=critic_nn).to(self.device)

        self.actor.optimizer = Adam(params=self.actor.parameters(), lr=actor_lr)
        self.critic.optimizer = Adam(params=self.critic.parameters(), lr=critic_lr)

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

    def setup_wandb_logger(self, config: dict, tags: list, disabled: bool = False):
        
        wdb_logger = WandbLogger(tags=tags)

        wdb_logger.disable_logging(disable=disabled)

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

        if self.extended_obs:

            state = self.get_extended_observations(state)

        # Launch rollout

        while not done:

            # Start by appending the state to create the states trajectory
            states.append(state)

            # Perform action and pass to next state

            actions, log_prob = self.select_action(tensor(state).float().to(self.device))

            state, reward, done, _ = self.env.step(actions)

            if self.extended_obs:

                state = self.get_extended_observations(state)

            rewards.append(reward)
            log_probs.append(log_prob)
            actions_hist.append(actions)

        rewards = np.stack(rewards)
        actions_hist = np.stack(actions_hist).squeeze(axis=-1)

        return states, rewards, log_probs, actions_hist

    def train(self, training_steps: int = 1000, min_loss: float = 0.01):

        all_states, all_rewards, all_actions, all_net_energy = [], [], [], []

        for step in tqdm(range(self.current_step, training_steps)):

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
            value = self.critic(states).squeeze(dim=-1)

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

            # Check stop condition

            stop_condition = actor_loss.abs().item() <= min_loss and critic_loss.abs().item() <= min_loss
            
            if step % 50 == 0 or stop_condition:

                # Wandb logging

                results = {
                    "rollout_avg_reward": rewards.mean(axis=1).sum(axis=0)[0],
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

        return all_states, all_rewards, all_actions, all_net_energy

    # Save weights to file

    def save_weights(self, actor_state_dict, actor_opt_state_dict, critic_state_dict, critic_opt_state_dict, current_step):

        model_path = self.wdb_logger.run.dir if self.wdb_logger.run is not None else './models'

        torch.save({
            'current_step': current_step,
            'actor_state_dict': actor_state_dict,
            'actor_opt_state_dict': actor_opt_state_dict,
            'critic_state_dict': critic_state_dict,
            'critic_opt_state_dict': critic_opt_state_dict,
        }, f'{model_path}/d_a2c_c_model.pt')

        print(f'Saving model on step: {current_step}')


"""
    Main method definition
"""

if __name__ == '__main__':

    config = load_config("d_a2c.yaml")
    config = config['train']
    
    # Read arguments from command line

    parser = argparse.ArgumentParser(prog='rl', description='RL Experiments')

    args = parser.parse_args([])

    parser.add_argument("-y", "--yaml", default=True, help="Load params from yaml file")
    parser.add_argument("-dl", "--disable_logging", default=False, action="store_true", help="Disable logging")
    parser.add_argument("-bs", "--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("-ts", "--training_steps", default=500, type=int, help="Steps for training loop")
    parser.add_argument("-rs", "--rollout_steps", default=8759, type=int, help="Steps for the rollout loop")
    parser.add_argument("-alr", "--actor_lr", default=1e-3, type=float, help="Actor learning rate")
    parser.add_argument("-clr", "--critic_lr", default=1e-3, type=float, help="Critic learning rate")
    parser.add_argument("-ann", "--actor_nn", default=256, type=int, help="Actor hidden layer number of neurons")
    parser.add_argument("-cnn", "--critic_nn", default=256, type=int, help="Critic hidden layer number of neurons")
    parser.add_argument("-nda", "--num_disc_act", default=40, type=int, help="Number of Discrete Actions")
    parser.add_argument("-g", "--gamma", default=0.95, type=float, help="Critic hidden layer number of neurons")
    parser.add_argument("-gpu", "--enable_gpu", default=False, action="store_true", help="Device to use for training")
    parser.add_argument("-ca", "--central_agent", default=False, action="store_true", help="Central agent")
    parser.add_argument("-rss", "--random_soc_0", default=False, action="store_true", help="Random starting soc")
    parser.add_argument("-dn", "--disable_noise", default=False, action="store_true", help="Disable noise from data generation")
    parser.add_argument("-e", "--encoding", default=False, action="store_true", help="Enable encoding")
    parser.add_argument("-xobs", "--extended_observation", default=False, action="store_true", help="Extended observation")

    args = parser.parse_args()

    # Get arguments from command line

    use_yaml = args.yaml
    
    if use_yaml:

        print('Run yaml')

        disable_logging = config['disable_logging']
        batch_size = config['batch_size']
        training_steps = config['training_steps']
        rollout_steps = config['rollout_steps']
        actor_lr = config['actor_lr']
        critic_lr = config['critic_lr']
        actor_nn = config['actor_nn']
        critic_nn = config['critic_nn']
        gamma = config['gamma']
        enable_gpu = config['enable_gpu']
        central_agent = config['central_agent']
        random_soc_0 = config['random_soc_0']
        encoding = config['encoding']
        extended_observation = config['extended_observation']
        disable_noise = config['disable_noise']
        num_disc_act = config['num_disc_act']

    else:
        
        print('Use params')

        disable_logging = args.disable_logging
        batch_size = args.batch_size
        training_steps = args.training_steps
        rollout_steps = args.rollout_steps
        actor_lr = args.actor_lr
        critic_lr = args.critic_lr
        actor_nn = args.actor_nn
        critic_nn = args.critic_nn
        gamma = args.gamma
        enable_gpu = args.enable_gpu
        central_agent = args.central_agent
        random_soc_0 = args.random_soc_0
        encoding = args.encoding
        extended_observation = args.extended_observation
        disable_noise = args.disable_noise
        num_disc_act = args.num_disc_act
    


    # Start wandb logger

    try:

        '''
            Setup all the configurations for Wandb
        '''

        wdb_config={
            "training_steps": training_steps,
            "batch_size": batch_size,
            "rollout_steps": rollout_steps,
            "agent_actor_lr": actor_lr,
            "agent_critic_lr": critic_lr,
            "agent_actor_nn": actor_nn,
            "agent_critic_nn": critic_nn,
            "gamma": gamma,
            "central_agent": central_agent,
            "random_soc_0": random_soc_0,
            "encoding": encoding,
            "extended_observation": extended_observation,
            "num_disc_act": num_disc_act,
        }

        '''
            Run the simulator
        '''

        set_all_seeds(0)

        # Instantiate the environment

        my_env = MGSimple(
            batch_size=batch_size, steps = rollout_steps, min_temp = 29, max_temp = 31, peak_pv_gen = 1, peak_grid_gen = 1, peak_load = 1,
            random_soc_0=random_soc_0, disable_noise=disable_noise
        )

        # Instantiate the agent

        agent = Agent(
            env=my_env, critic_lr=critic_lr, actor_lr=actor_lr, actor_nn=actor_nn, critic_nn=critic_nn, batch_size=batch_size, gamma=gamma,
            extended_obs=extended_observation, wandb_dict=wdb_config, enable_gpu=enable_gpu, disable_wandb=disable_logging,num_disc_act=num_disc_act
        )

        # Launch the training

        agent.train(training_steps=training_steps)

        # Finish wandb process

        agent.wdb_logger.finish()

    except (RuntimeError, KeyboardInterrupt):

        traceback.print_exc()