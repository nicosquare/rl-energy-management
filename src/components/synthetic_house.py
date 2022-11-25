from typing import Union
import numpy as np

from matplotlib import pyplot as plt

from src.components.battery import Battery, BatteryParameters

class SyntheticHouse():
    
    def __init__(
        self, config
    ):
        
        self.batch_size = config['batch_size']
        self.steps = config['rollout_steps']
        self.peak_pv_gen = config['pv']['peak_pv_gen']
        self.peak_grid_gen = config['peak_grid_gen']
        self.peak_load = config['profile']['peak_load']
        self.grid_sell_rate = config['grid_sell_rate']
        self.disable_noise = config['disable_noise']
        self.profile = config['profile']['type']

        # Time variables

        self.time = np.arange(self.steps)
        self.current_step = 0

        # Environmental variables

        self.min_temp = config['min_temp']
        self.max_temp = config['max_temp']
        self.temp = np.random.uniform(self.min_temp, self.max_temp, self.steps)

        # Microgrid data

        self.pv_gen = None
        self.demand = None
        self.total_gen = None
        self.remaining_energy = None
        self.net_energy = None
        self.price = None
        self.emission = None

        # Components
        self.random_soc_0 = config['battery']['random_soc_0']
        self.battery = Battery(batch_size = self.batch_size, random_soc_0=self.random_soc_0, params = BatteryParameters(config['battery']))

        # Save house attributes as an array

        self.profile_types = ['family', 'business', 'teenagers']

        # Encode profile type like a one-hot vector # TODO: Improve the utils OneHotEncoder to accept strings

        self.attr = np.zeros(len(self.profile_types))
        self.attr[self.profile_types.index(config['profile']['type'])] = 1

        # self.attr = np.insert(self.attr, self.attr.shape[0], [
        #     config['profile']['peak_load'],
        #     config['battery']['capacity'],
            # config['battery']['efficiency'],
            # config['battery']['soc_max'],
            # config['battery']['soc_min'],
            # config['battery']['p_charge_max'],
            # config['battery']['p_discharge_max'],
        #     config['pv']['peak_pv_gen'],
        # ])

        # Generate data

        self.generate_data()
        

    def generate_data(self):

        min_noise_pv = 0
        max_noise_pv = 0.1
        min_noise_demand = 0
        max_noise_demand = 0.01

        if self.disable_noise:
    
            max_noise_pv = 0
            max_noise_demand = 0

        # Generate data

        # Solar Generation

        _, self.pv_gen = self.pv_generation(min_noise=min_noise_pv, max_noise=max_noise_pv)

        # Demand Profile

        if self.profile == 'family':
            _, self.demand = self.demand_family(min_noise=min_noise_demand, max_noise=max_noise_demand)
        elif self.profile == 'business':
            _, self.demand = self.demand_home_business(min_noise=min_noise_demand, max_noise=max_noise_demand)
        elif self.profile == 'teenagers':
            _, self.demand = self.demand_teenagers(min_noise=min_noise_demand, max_noise=max_noise_demand)
        else:
            _, self.demand = self.demand_family(min_noise=min_noise_demand, max_noise=max_noise_demand)
        
        # Grid electric production
        _, _, self.total_gen, self.price, self.emission = self.grid_price_and_emission(
            gas_price=0.5, nuclear_price=0.1, gas_emission_factor=0.9, nuclear_emission_factor=0.1
        )

            # Net energy without battery
        self.remaining_energy = self.demand - self.pv_gen

        # Net energy starts with remaining energy value as not action has been taken yet

        self.net_energy = np.zeros((self.batch_size, self.steps))

    def pv_generation(self, min_noise: float = 0, max_noise: float = 0.1):

        base = np.sin((self.time/4) + 5)
        noise = np.random.normal(min_noise, max_noise, self.steps)

        # Generation is clipped because it can't be negative

        gen = ((base + noise) * self.peak_pv_gen).clip(min=0)

        return base.clip(min=0), gen

    def demand_from_day_profile(
        self, day_profile: np.ndarray, base_power_rate: float = 0.2, min_noise: float = 0, max_noise: float = 0.01,
    ):
            
        base = np.ones(self.steps) * self.peak_load * base_power_rate

        # Generate a complete profile from a daily

        complete_profile = np.tile(day_profile, self.steps//len(day_profile)) * (self.peak_load - base.max())

        if len(complete_profile) < self.steps:
            complete_profile = np.concatenate((complete_profile, day_profile[:self.steps-len(complete_profile)]))

        # Generate external factors

        noise = np.random.normal(min_noise, max_noise, self.steps)

        # Generate demand

        full_base = base + complete_profile
        demand = full_base + noise + (self.temp - 22) * 0.01

        return full_base, demand

    def demand_family(self, min_noise: float = 0, max_noise: float = 0.06):

        # Day profile defined arbitrarily according to the assumed behaviour of a family

        day_profile = np.array([
            0, 0, 0, 0, 0, 0.25, 0.83, 1, 0.25, 0, 0, 0, 0, 0, 0, 0.33, 0.41, 0.41, 0.66, 0.83, 0.66, 0.25, 0.08, 0
        ])

        return self.demand_from_day_profile(day_profile, base_power_rate=0.2, min_noise=min_noise, max_noise=max_noise)

    def demand_teenagers(self, min_noise: float = 0, max_noise: float = 0.06):

        # Day profile defined arbitrarily according to the assumed behaviour of teenagers

        day_profile = np.array([
            1, 1, 1, 0.83, 0.41, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.41, 0.41, 0.41, 0.41, 0.41, 0.83, 1, 1, 1, 1
        ])

        return self.demand_from_day_profile(day_profile, base_power_rate=0.3, min_noise=min_noise, max_noise=max_noise)

    def demand_home_business(self, min_noise: float = 0, max_noise: float = 0.06):
            
        # Day profile defined arbitrarily according to the assumed behaviour of a home business

        day_profile = np.array([
            0, 0, 0, 0, 0, 0.25, 0.83, 1, 0.25, 0, 0, 0, 0, 0, 0, 0.33, 0.41, 0.41, 0.66, 0.83, 0.66, 0.25, 0.83, 0
        ])

        return self.demand_from_day_profile(day_profile, base_power_rate=0.6, min_noise=min_noise, max_noise=max_noise)

    def grid_price_and_emission(
        self, nuclear_energy_rate: float = 0.6, nuclear_price: float = 0.1, nuclear_emission_factor: float = 0.01,
        gas_price: float = 0.3, gas_emission_factor: float = 0.2
    ):
        
        # Assume a mix between nuclear and gas power plants

        nuclear_gen = np.ones(self.steps) * nuclear_energy_rate * self.peak_grid_gen
        daily_gas_gen = np.array([
            0, 0, 0, 0, 0, 0, 0.3, 0.6, 1, 0.6, 0.3, 0.3, 0.3, 0.3, 0.6, 1, 0.6, 0.3, 0, 0, 0, 0, 0, 0
        ]) * (self.peak_grid_gen - nuclear_gen.max())

        # Generate a complete profile from a daily

        gas_gen = np.tile(daily_gas_gen, self.steps//len(daily_gas_gen))

        if len(gas_gen) < self.steps:
            gas_gen = np.concatenate((gas_gen, daily_gas_gen[:self.steps-len(gas_gen)]))

        total_gen = nuclear_gen + gas_gen

        # Compute price and emission

        price = nuclear_gen * nuclear_price + gas_gen * gas_price
        emission = nuclear_gen * nuclear_emission_factor + gas_gen * gas_emission_factor

        return nuclear_gen, gas_gen, total_gen, price, emission

    def compute_metrics(self):

        remaining_energy_to_step = self.remaining_energy[:self.current_step]

        # Compute battery performance

        price_without_battery = np.where(
            remaining_energy_to_step > 0,
            remaining_energy_to_step * self.price,
            remaining_energy_to_step * self.price * self.grid_sell_rate
        ).sum()

        price_with_battery = np.where(
            self.net_energy > 0,
            self.net_energy * self.price,
            self.net_energy * self.price * self.grid_sell_rate
        ).sum(axis=1).mean()

        emission_without_battery = np.where(
            remaining_energy_to_step > 0,
            remaining_energy_to_step * self.emission,
            0
        ).sum()

        emission_with_battery = np.where(
            self.net_energy > 0,
            self.net_energy * self.emission,
            0
        ).sum(axis=1).mean()

        # Compute metrics, we add a baseline of one to indicate when the battery didn't cause any improvement

        price_metric = price_with_battery - price_without_battery
        emission_metric = emission_with_battery - emission_without_battery

        return price_metric, emission_metric

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

        p_charge, p_discharge, _ = self.battery.check_battery_constraints(power_rate=batt_action)
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
