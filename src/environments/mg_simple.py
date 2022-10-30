import numpy as np

from gym import Env
from gym.spaces import Box

from src.components.microgrid_simple import SimpleMicrogrid

class MGSimple(Env):

    def __init__(self, batch_size: int = 1, steps: int = 8760, min_temp: float = 29, max_temp: float = 31, peak_pv_gen: int = 1, peak_conv_gen: float = 1, peak_load: float = 1):
        
        """
        Gym environment to simulate a simple Microgrid scenario
        """

        """
        Observation space is composed by:
        
            hour_of_day: [0, 23]
            temperature: [29, 31]
            pv_generation: [0, 1]
            demand: [0, 1]
            net_energy: [-1, 1]
            grid_sell_price: [0, 1]
            grid_buy_price: [0, 1]
            grid_emission_factor: [0, 1]
            soc: [0,1]
        
        """

        self.batch_size = batch_size

        self.observation_space = Box(
            low=np.float32(np.array([0.0, 29.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0])),
            high=np.float32(np.array([23.0, 31.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
            shape=(8,),
            dtype=np.float32
        )

        """
        Action space is composed by:
            batt_action: [-1, 1]
        """

        self.action_space = Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        self.mg = SimpleMicrogrid(
            batch_size=batch_size, steps=steps, min_temp=min_temp, max_temp=max_temp, peak_pv_gen=peak_pv_gen, peak_conv_gen=peak_conv_gen, peak_load=peak_load
        )

    def observe(self):
        return self.mg.observe()

    def step(self, action: np.ndarray):

        state, reward = self.mg.apply_action(batt_action=action)

        state = state
        done = self.mg.current_step >= (self.mg.steps - 1)
        info = {}

        return state, reward, done, info

    def reset(self):
        self.mg.reset()
        return self.observe(), np.zeros((self.batch_size, 1)), False, {}

    def render(self, mode="human"):
        print('Rendering not defined yet')