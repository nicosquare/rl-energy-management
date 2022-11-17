import numpy as np
from typing import Union, List

from src.components.synthetic_house import SyntheticHouse

class SyntheticMicrogrid():
    
    def __init__(
        self, config
    ):
        
        self.batch_size = config['batch_size']
        self.steps = config['rollout_steps']
        self.peak_grid_gen = config['peak_grid_gen']
        self.grid_sell_rate = config['grid_sell_rate']
        self.disable_noise = config['disable_noise']

        # Time variables

        self.time = np.arange(self.steps)
        self.current_step = 0

        # Environmental variables

        self.min_temp = config['min_temp']
        self.max_temp = config['max_temp']
        self.temp = np.random.uniform(self.min_temp, self.max_temp, self.steps)

        # Houses
        
        self.houses = self.house_loader(config["houses"])

    @property
    def net_energy(self):
        return np.stack([house.net_energy for house in self.houses])

    def house_loader(self, config) -> List[SyntheticHouse]:

        houses = []

        for _, attr in zip(config, config.values()):

            config = attr

            # Append necessary information for SyntheticHouse class

            config['batch_size'] = self.batch_size
            config['rollout_steps'] = self.steps
            config['peak_grid_gen'] = self.peak_grid_gen
            config['grid_sell_rate'] = self.grid_sell_rate
            config['disable_noise'] = self.disable_noise
            config['min_temp'] = self.min_temp
            config['max_temp'] = self.max_temp

            # Create each house instance

            houses.append(SyntheticHouse(config=config))
        
        return houses

    def observe(self) -> np.ndarray:

        return np.stack([house.observe() for house in self.houses], axis=0)

    def apply_action(self, batt_action: np.array) -> Union[np.ndarray, np.ndarray]:

        next_state = []
        reward = []

        # Apply the corresponding action to each house

        for index, house in enumerate(self.houses):

            house_obs, house_reward = house.apply_action(batt_action=batt_action[index])

            next_state.append(house_obs)
            reward.append(house_reward)

        self.increment_step()

        return np.stack(next_state, axis=0), np.stack(reward, axis=0)

    def get_houses_attrs(self) -> np.ndarray:

        return np.stack([house.attr for house in self.houses], axis=0)

    def increment_step(self) -> None:

        # The current_step of the microgrid is the same for any of the houses

        self.current_step = self.houses[0].current_step

    def reset(self):
        
        for house in self.houses:
            house.reset()

        self.current_step = self.houses[0].current_step