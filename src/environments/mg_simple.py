import numpy as np

from gym import Env
from gym.spaces import Box

from src.components.microgrid_simple import SimpleMicrogrid

class MGSimple(Env):

    def __init__(
        self, batch_size: int = 1, steps: int = 8760, min_temp: float = 29, max_temp: float = 31, peak_pv_gen: int = 1, peak_grid_gen: float = 1, peak_load: float = 1,
        grid_sell_rate: float = 0.25, disable_noise: bool = False, random_soc_0: bool = False
    ):
        
        """
        Gym environment to simulate a simple Microgrid scenario
        """

        """
        Observation space is composed by:
        
            0 hour_of_day: [0, 23]
            1 temperature: [29, 31]
            2 pv_generation_t: [0, 1]
            3 pv_generation_t+1: [0, 1]
            4 pv_generation_t+2: [0, 1]
            5 pv_generation_t+3: [0, 1]
            6 pv_generation_t+4: [0, 1]
            7 pv_generation_t+5: [0, 1]
            8 pv_generation_t+6: [0, 1]
            9 demand_t: [0, 1]
            10 demand_t+1: [0, 1]
            11 demand_t+2: [0, 1]
            12 demand_t+3: [0, 1]
            13 demand_t+4: [0, 1]
            14 demand_t+5: [0, 1]
            15 demand_t+6: [0, 1]
            16 grid_sell_price: [0, 1]
            17 grid_buy_price: [0, 1]
            18 grid_emission_factor: [0, 1]
            19 soc: [0,1]
        
        """

        self.batch_size = batch_size

        low_limit_obs = np.float32(np.array([
            0.0, 29.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]))
        high_limit_obs = np.float32(np.array([
            23.0, 31.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]))

        self.observation_space = Box(
            low=low_limit_obs,
            high=high_limit_obs,
            shape=low_limit_obs.shape,
            dtype=np.float32
        )

        """
        Action space is composed by:
            batt_action: [-1, 1]
        """

        low_limit_action = np.float32(np.array([-1.0]))
        high_limit_action = np.float32(np.array([1.0]))

        self.action_space = Box(
            low=low_limit_action,
            high=high_limit_action,
            shape=low_limit_action.shape,
            dtype=np.float32
        )

        self.mg = SimpleMicrogrid(
            batch_size=batch_size, steps=steps, min_temp=min_temp, max_temp=max_temp, peak_pv_gen=peak_pv_gen, peak_grid_gen=peak_grid_gen, peak_load=peak_load,
            grid_sell_rate= grid_sell_rate, disable_noise=disable_noise, random_soc_0=random_soc_0
        )

    def observe(self):
        return self.mg.observe()

    def step(self, action: np.ndarray):

        done = self.mg.current_step >= self.mg.steps
        info = {}
        
        if not done:
            
            reward = self.mg.apply_action(batt_action=action)
            state = self.mg.observe()

            self.mg.increment_step()
            
        else:
                
            state = np.zeros(self.observation_space.shape)
            reward = np.zeros((self.batch_size, 1))

        return state, reward, done, info

    def reset(self):
        self.mg.reset()
        return self.observe(), np.zeros((self.batch_size, 1)), False, {}

    def render(self, mode="human"):
        print('Rendering not defined yet')
