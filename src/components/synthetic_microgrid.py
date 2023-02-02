import numpy as np
from typing import Union, List
from itertools import cycle

from src.components.synthetic_house import SyntheticHouse

class SyntheticMicrogrid():
    
    def __init__(
        self, config,
    ):
        
        self.batch_size = config['batch_size']
        self.steps = config['rollout_steps']
        self.grid_profiles = cycle(config['grid']['profiles'].values())
        self.current_profile = next(self.grid_profiles)
        self.disable_noise = config['disable_noise']
        self.houses_config = config['houses']

        # Time variables

        self.time = np.arange(self.steps)
        self.current_step = 0

        # Environmental variables

        self.min_temp = config['min_temp']
        self.max_temp = config['max_temp']
        self.temp = np.random.uniform(self.min_temp, self.max_temp, self.steps)

        # Houses, by default they are loaded in train mode
        
        self.houses = self.house_loader(mode='train')

    @property
    def net_energy(self):
        return np.stack([house.net_energy for house in self.houses])

    def house_loader(self, mode : str = 'train') -> List[SyntheticHouse]:

        houses = []
        mode_config = self.houses_config[mode]

        for _, attr in zip(mode_config, mode_config.values()):

            house_config = attr

            # Append necessary information for SyntheticHouse class

            house_config['batch_size'] = self.batch_size
            house_config['rollout_steps'] = self.steps
            house_config['grid_profile'] = self.current_profile
            house_config['disable_noise'] = self.disable_noise
            house_config['min_temp'] = self.min_temp
            house_config['max_temp'] = self.max_temp

            # Create each house instance

            houses.append(SyntheticHouse(config=house_config))
        
        return houses

    def change_grid_profile(self):
        
        self.current_profile = next(self.grid_profiles)

        for house in self.houses:
            house.change_grid_profile(profile=self.current_profile)

    def change_mode(self, mode: str):
        
        self.houses = self.house_loader(mode=mode)

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

    def get_houses_metrics(self) -> np.ndarray:

        perf = np.stack([house.compute_metrics() for house in self.houses], axis=0)
        price_perf = perf[:,0]
        emissions_perf = perf[:,1]

        return price_perf, emissions_perf

    def get_houses_attrs(self) -> np.ndarray:

        return np.stack([house.attr for house in self.houses], axis=0)

    def increment_step(self) -> None:

        # The current_step of the microgrid is the same for any of the houses

        self.current_step = self.houses[0].current_step

    def reset(self):
        
        for house in self.houses:
            house.reset()

        self.current_step = self.houses[0].current_step