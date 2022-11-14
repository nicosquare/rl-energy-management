import numpy as np
from typing import Union
from matplotlib import pyplot as plt

from src.components.synthetic_house import SyntheticHouse

class SyntheticCommunity():
    
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

        # Generate data
        self.generate_data()

    def house_loader(self, config):

        houses = []

        for _, attr in zip(config, config.values()):

            config = attr
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

        terminal = self.current_step >= self.steps

        if not terminal:

            state = np.stack([
                np.ones(self.batch_size) * self.current_step % 24,
                self.battery.soc.squeeze(axis=-1)
            ], axis=1)

        else:

            state = np.stack([
                np.zeros(self.batch_size),
                np.ones(self.batch_size) * self.battery.soc_min
            ], axis=1)

        return state

    def apply_action(self, batt_action: np.array) -> Union[np.ndarray, np.ndarray]:

        # Apply action to battery and reach the new state

        p_charge, p_discharge, i_action = self.battery.check_battery_constraints(power_rate=batt_action)
        self.battery.apply_action(p_charge = p_charge, p_discharge = p_discharge)

        # Compute the next step net energy

        self.net_energy[:,self.current_step] += (self.remaining_energy[self.current_step] + p_charge - p_discharge).squeeze()

        # Compute cost

        cost = np.where(
            self.net_energy[:,self.current_step] > 0,
            (self.net_energy[:,self.current_step]) * (self.price[self.current_step] + self.emission[self.current_step]),
            (self.net_energy[:,self.current_step]) * self.price[self.current_step] * self.grid_sell_rate
        ).reshape(self.batch_size,1)
        
        self.increment_step()

        return self.observe(), -cost

    def increment_step(self) -> None:
        self.current_step += 1

    def reset(self):
        """
            Resets the current time step.
        Returns
        -------
            None
        """
        self.current_step = 0
        self.generate_data()
        self.battery.reset()
