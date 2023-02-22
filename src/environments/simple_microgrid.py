import numpy as np

from gym import Env
from gym.spaces import Box

from src.utils.preprocessing import OneHotEncoding, NoNormalization
from src.components.synthetic_microgrid import SyntheticMicrogrid

class SimpleMicrogrid(Env):

    def __init__(
        self, config
    ):
        
        """
        Gym environment to simulate a simple Microgrid scenario
        """

        """
        Observation space is composed by:
        
            0 hour_of_day: [0, 23]
            1 soc: [0,1]
        
        """

        # Set up microgrid

        self.mg = SyntheticMicrogrid(config=config)

        # Get params from yaml config file
        self.batch_size = config['batch_size']
        self.encoding = config['encoding']

        # Build a multibatch house attr array

        self.houses_attr = np.repeat(self.mg.get_houses_attrs()[:, np.newaxis, :], self.batch_size, axis=1)
        self.house_attr_size = self.houses_attr.shape[2]

        self.n_houses = len(self.mg.houses)

        # Build a multibatch grid attr array

        self.grid_attr = np.repeat(np.repeat(self.mg.attr[np.newaxis,np.newaxis,:], self.batch_size, axis=1), self.n_houses, axis=0)
        self.grid_attr_size = self.grid_attr.shape[2]

        # Complete attribute array

        self.attr = np.concatenate((self.houses_attr, self.grid_attr), axis=2)
        self.attr_size = self.attr.shape[2]

        # Define environment characteristics

        low_limit_obs = np.float32(np.array([0.0, 0.0]))
        high_limit_obs = np.float32(np.array([23.0, 1.0]))

        self.encoders = [
            OneHotEncoding(range(24)), # Hour of day
            NoNormalization() # SOC
        ]

        self.observation_space = Box(
            low=low_limit_obs,
            high=high_limit_obs,
            shape=low_limit_obs.shape,
            dtype=np.float32
        )

        # Get the real size of the observation with the encoding

        self.obs_size = self.normalize_obs(obs=np.array([[low_limit_obs]])).shape[2]

        """
        Action space is composed by:

            One batt_action: [-1, 1] per house
        
        """

        low_limit_action = (np.ones((1,self.n_houses)) * -1).astype(np.float32)
        high_limit_action = (np.ones((1,self.n_houses))).astype(np.float32)

        self.action_space = Box(
            low=low_limit_action,
            high=high_limit_action,
            shape=low_limit_action.shape,
            dtype=np.float32
        )

    def observe(self):
        return self.normalize_obs(self.mg.observe())

    def step(self, action: np.ndarray):

        state = self.mg.apply_action(batt_action=action)
        reward = self.mg.compute_reward()
        state = self.normalize_obs(state)
        done = self.mg.current_step >= self.mg.steps
        info = {}
            
        return state, reward, done, info

    def reset(self):
        self.mg.reset()
        return self.observe(), np.zeros((self.mg.batch_size, 1)), False, {}

    def render(self, mode="human"):
        print('Rendering not defined yet')

    def normalize_obs(self, obs):

        encoded_obs = []

        if self.encoding:

            for i in range(len(obs)):

                house_encoded_obs = None

                for j, encoder in enumerate(self.encoders):
        
                    if house_encoded_obs is None:
                        house_encoded_obs = encoder*obs[i,:,j]
                    else:
                        house_encoded_obs = np.insert(house_encoded_obs, house_encoded_obs.shape[1], encoder*obs[i,:,j], axis=1)

                encoded_obs.append(house_encoded_obs)

            encoded_obs = np.stack(encoded_obs, axis=0)

        else:

            encoded_obs = obs

        return encoded_obs

    def change_mode(self, mode: str):

        self.mg.change_mode(mode)
        
        # Update houses related information

        self.houses_attr = np.repeat(self.mg.get_houses_attrs()[:, np.newaxis, :], self.batch_size, axis=1)
        self.house_attr_size = self.houses_attr.shape[2]
        self.n_houses = len(self.mg.houses)

        # Update grid related information

        self.grid_attr = np.repeat(np.repeat(self.mg.attr[np.newaxis,np.newaxis,:], self.batch_size, axis=1), self.n_houses, axis=0)
        self.grid_attr_size = self.grid_attr.shape[2]

        # Update the environment attribute array

        self.attr = np.concatenate((self.houses_attr, self.grid_attr), axis=2)
        self.attr_size = self.attr.shape[2]

        