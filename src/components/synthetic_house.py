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
        self.l3_export_rate = config['l3_export_rate']
        self.l3_import_fraction = config['l3_import_fraction']
        self.l3_emission = config['l3_emission']
        self.peak_pv_gen = config['pv']['peak_pv_gen']
        self.peak_load = config['profile']['peak_load']
        self.disable_noise = config['disable_noise']
        self.profile = config['profile']['type']

        # Time variables

        self.time = np.arange(self.steps)
        self.current_step = 0

        # Environmental variables

        self.min_temp = config['min_temp']
        self.max_temp = config['max_temp']
        #TODO change random generation to match hour/env factors
        self.temp = np.random.uniform(self.min_temp, self.max_temp, self.steps)

        # Microgrid data

        self.pv_gen = None
        self.demand = None
        self.remaining_energy = None
        self.net_energy = None

        # Import related registers

        self.l1_import_no_batt = np.zeros(self.steps)
        self.l1_import_rate_no_batt = np.zeros(self.steps)
        
        # Export related registers

        self.l1_export_no_batt = np.zeros(self.steps)
        self.l1_export_rate_no_batt = np.zeros(self.steps)

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
        self.initialize_registers()
        

    def generate_data(self):

        min_noise_pv = 0
        max_noise_pv = 0.1
        min_noise_demand = 0
        max_noise_demand = 0.01

        if self.disable_noise:
    
            max_noise_pv = 0
            max_noise_demand = 0

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

        # Net energy without battery
        
        self.remaining_energy = self.demand - self.pv_gen
        self.net_energy_no_batt = self.remaining_energy.copy()

    def initialize_registers(self):

        # Net energy starts with remaining energy value as not action has been taken yet

        self.net_energy = np.zeros((self.batch_size, self.steps))

        # Import related registers

        self.l1_import = np.zeros((self.batch_size, self.steps))
        self.l1_import_rate = np.zeros((self.batch_size, self.steps))
        
        # Export related registers

        self.l1_export = np.zeros((self.batch_size, self.steps))
        self.l1_export_rate = np.zeros((self.batch_size, self.steps))

    def pv_generation(self, min_noise: float = 0, max_noise: float = 0.1):

        base = np.sin((self.time/4) + 5)
        noise = np.random.normal(min_noise, max_noise, self.steps)

        # Generation is clipped because it can't be negative

        gen = ((base + noise) * self.peak_pv_gen).clip(min=0)

        # Normalize generation
        
        gen = gen / gen.max() if gen.max() > 0 else gen

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
        demand = full_base + noise + np.abs(self.temp - 22) * 0.01

        # Normalize demand

        demand = demand / demand.max() if demand.max() > 0 else demand

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
            0, 0, 0, 0, 0, 0.85, 0.83, 1, 0.85, 0.8, 0.6, 0.6, 0.82, 0.7, 0.8, 0.63, 0.61, 0.41, 0.46, 0.43, 0.16, 0.25, 0.23, 0
        ])

        return self.demand_from_day_profile(day_profile, base_power_rate=0.6, min_noise=min_noise, max_noise=max_noise)

    def compute_metrics(self):

        price_without_battery, emission_without_battery = self.compute_metrics_no_batt(step=self.current_step)

        # Compute battery performance

        price_with_battery = np.where(
            self.net_energy > 0,
            self.net_energy * self.l3_export_rate + self.l1_import * self.l1_import_rate,
            self.net_energy * self.l3_export_rate * self.l3_import_fraction - self.l1_export * self.l1_export_rate + self.l1_import * self.l1_import_rate
        ).sum(axis=1).mean()

        emission_with_battery = np.where(
            self.net_energy > 0,
            self.net_energy * self.l3_emission,
            0
        ).sum(axis=1).mean()

        # Compute metrics, we add a baseline of one to indicate when the battery didn't cause any improvement

        price_metric = price_with_battery - price_without_battery
        emission_metric = emission_with_battery - emission_without_battery

        return price_metric, emission_metric

    def compute_metrics_no_batt(self, step: int = None):

        net_energy_no_batt_to_step = self.net_energy_no_batt[:step]

        # Compute battery performance

        price_without_battery = np.where(
            net_energy_no_batt_to_step > 0,
            net_energy_no_batt_to_step * self.l3_export_rate + self.l1_import_no_batt * self.l1_import_rate_no_batt,
            net_energy_no_batt_to_step * self.l3_export_rate * self.l3_import_fraction - self.l1_export_no_batt * self.l1_export_rate_no_batt + self.l1_import_no_batt * self.l1_import_rate_no_batt
        ).sum()

        emission_without_battery = np.where(
            net_energy_no_batt_to_step > 0,
            net_energy_no_batt_to_step * self.l3_emission,
            0
        ).sum()

        return price_without_battery, emission_without_battery

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

        return self.observe()

    def compute_reward(self) -> np.ndarray:

        # Compute reward

        cost = np.where(
            self.net_energy[:,self.current_step] > 0,
            # If we are buying energy to the grid or L1 houses
                self.net_energy[:,self.current_step] * (self.l3_export_rate[self.current_step]) +
                self.l1_import[:,self.current_step] * self.l1_import_rate[:,self.current_step]
            ,
            # If we are selling energy to the grid or L1 houses
                self.net_energy[:,self.current_step] * self.l3_export_rate[self.current_step] * self.l3_import_fraction +
                self.l1_import[:,self.current_step] * self.l1_import_rate[:,self.current_step] -
                self.l1_export[:,self.current_step] * self.l1_export_rate[:,self.current_step]
        ).reshape(self.batch_size,1)

        self.increment_step()

        return -cost

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
        self.initialize_registers()
        self.battery.reset()
