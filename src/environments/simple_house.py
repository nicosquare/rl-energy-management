import numpy as np

from gym import Env
from gym.spaces import Box

from src.utils.preprocessing import OneHotEncoding, NoNormalization
from src.components.synthetic_house import SyntheticHouse

class SimpleHouse(Env):

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
        # Get params from yaml config file
        self.encoding = config['encoding']


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

        self.obs_size = self.normalize_obs(obs=np.array([low_limit_obs])).shape[1]

        """
        Action space is composed by:

            0 batt_action: [-1, 1]
        
        """

        low_limit_action = np.float32(np.array([-1.0]))
        high_limit_action = np.float32(np.array([1.0]))

        self.action_space = Box(
            low=low_limit_action,
            high=high_limit_action,
            shape=low_limit_action.shape,
            dtype=np.float32
        )

        self.mg = SyntheticHouse(config=config)

    def observe(self):
        return self.normalize_obs(self.mg.observe())

    def step(self, action: np.ndarray):

        state, reward = self.mg.apply_action(batt_action=action)
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

        encoded_obs = None

        if self.encoding:

            for i, encoder in enumerate(self.encoders):

                if encoded_obs is None:
                    encoded_obs = encoder*obs[:,i]
                else:
                    encoded_obs = np.insert(encoded_obs, encoded_obs.shape[1], encoder*obs[:,i], axis=1)
        else:

            encoded_obs = obs

        return encoded_obs
    