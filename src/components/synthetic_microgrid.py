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
        self.profile_types = [c['name'] for c in config['grid']['profiles'].values()]
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

        # Microgrid data

        self.price = None
        self.emission = None
        
        # Houses, by default they are loaded in train mode
        
        self.houses = self.house_loader(mode='train')
        self.compute_transactions_without_batt()

        # Grid attributes

        self.attr = self.encode_grid_attributes()

    @property
    def net_energy(self):
        return np.stack([house.net_energy for house in self.houses]).mean(axis=1)
    
    @property
    def l1_export(self):
        return np.stack([house.l1_export for house in self.houses]).mean(axis=1)
    
    @property
    def l1_import(self):
        return np.stack([house.l1_import for house in self.houses]).mean(axis=1)
    
    @property
    def l1_export_rate_no_batt(self):
        return self.houses[0].l1_export_rate_no_batt
    
    @property
    def l1_import_rate_no_batt(self):
        return self.houses[0].l1_export_rate_no_batt
    
    @property
    def l1_export_rate(self):
        return np.stack([house.l1_export_rate for house in self.houses]).mean(axis=1)
    
    @property
    def l1_import_rate(self):
        return np.stack([house.l1_import_rate for house in self.houses]).mean(axis=1)
    
    @property
    def soc(self):
        return np.stack([house.battery.hist_soc for house in self.houses]).mean(axis=1)
    
    @property
    def p_charge(self):
        return np.stack([house.battery.hist_p_charge for house in self.houses]).mean(axis=1)
    
    @property
    def p_discharge(self):
        return np.stack([house.battery.hist_p_discharge for house in self.houses]).mean(axis=1)

    @property
    def offer(self):
        return np.maximum(-self.net_energy.mean(axis=0),0)
    
    @property
    def demand(self):
        return np.maximum(self.net_energy.mean(axis=0),0)

    def encode_grid_attributes(self):

        # Encode profile type like a one-hot vector # TODO: Improve the utils OneHotEncoder to accept strings

        attr = np.zeros(len(self.profile_types))
        attr[self.profile_types.index(self.current_profile['name'])] = 1

        return attr

    def house_loader(self, mode : str = 'train') -> List[SyntheticHouse]:

        houses = []
        mode_config = self.houses_config[mode]
        
        # Compute grid price, l1 rates and emission

        self.price, self.emission = self.grid_price_and_emission()

        for _, attr in zip(mode_config, mode_config.values()):

            house_config = attr

            # Append necessary information for SyntheticHouse class

            house_config['batch_size'] = self.batch_size
            house_config['rollout_steps'] = self.steps
            house_config['l3_export_rate'] = self.price
            house_config['l3_import_fraction'] = self.current_profile['import_fraction']
            house_config['l3_emission'] = self.emission
            house_config['disable_noise'] = self.disable_noise
            house_config['min_temp'] = self.min_temp
            house_config['max_temp'] = self.max_temp

            # Create each house instance

            houses.append(SyntheticHouse(config=house_config))
        
        return houses

    def grid_price_and_emission(self):

        peak_grid_gen = self.current_profile['peak_gen']
        nuclear_energy_rate = self.current_profile['nuclear_energy_rate']
        nuclear_price = self.current_profile['nuclear_price']
        nuclear_emission_factor = self.current_profile['nuclear_emission_factor']
        gas_price = self.current_profile['gas_price']
        gas_emission_factor = self.current_profile['gas_emission_factor']
        gas_profile = np.array(self.current_profile['gas_profile'])
        
        # Assume a mix between nuclear and gas power plants

        nuclear_gen = np.ones(self.steps) * nuclear_energy_rate * peak_grid_gen
        daily_gas_gen = gas_profile * (peak_grid_gen - nuclear_gen[0])

        # Generate a complete profile from a daily

        gas_gen = np.tile(daily_gas_gen, self.steps//len(daily_gas_gen))

        if len(gas_gen) < self.steps:
            gas_gen = np.concatenate((gas_gen, daily_gas_gen[:self.steps-len(gas_gen)]))

        # Compute price, l1 rates and emission

        price = nuclear_gen * nuclear_price + gas_gen * gas_price
        emission = nuclear_gen * nuclear_emission_factor + gas_gen * gas_emission_factor
        
        return price, emission

    def change_grid_profile(self):
        
        self.current_profile = next(self.grid_profiles)
        self.attr = self.encode_grid_attributes()

        self.price, self.emission = self.grid_price_and_emission()

        for house in self.houses:
            house.l3_export_rate = self.price
            house.l3_emission = self.emission
            house.l3_import_fraction = self.current_profile['import_fraction']

        self.compute_transactions_without_batt()

    def change_mode(self, mode: str):
        
        self.houses = self.house_loader(mode=mode)
        self.compute_transactions_without_batt()

    def observe(self) -> np.ndarray:

        return np.stack([house.observe() for house in self.houses], axis=0)

    def apply_action(self, batt_action: np.array) -> Union[np.ndarray, np.ndarray]:

        next_state = []

        # Apply the corresponding action to each house

        for index, house in enumerate(self.houses):

            house_obs = house.apply_action(batt_action=batt_action[index])

            next_state.append(house_obs)

        return np.stack(next_state, axis=0)

    def compute_transactions_without_batt(self) -> None:

        # Compute the net energy of each house and the surplus/shortage

        mg_remaining_energy = np.stack([house.remaining_energy for house in self.houses], axis=1)
        mg_surplus = np.maximum(0,-mg_remaining_energy)
        mg_shortage = np.maximum(0,mg_remaining_energy)

        for step in range(self.steps):

            step_surplus_no_batt = mg_surplus[step]
            step_shortage_no_batt = mg_shortage[step]
            step_demand = np.sum(step_shortage_no_batt)
            step_offer = np.sum(step_surplus_no_batt)

            # Compute selling and buying rate according to supply and demand

            baseline = self.price[step] * (1 - self.current_profile['import_fraction'])
            l1_import_rate = baseline + self.current_profile['l1_alpha'] * step_demand - self.current_profile['l1_beta'] * step_offer
            l1_export_rate = l1_import_rate - self.current_profile['l1_fee']

            # Compute the energy transacted and the contribution of each house

            step_transacted_energy = np.minimum(step_offer, step_demand)

            # Update the houses for the current time step

            for house_ix, house in enumerate(self.houses):

                # Update the selling and buying rate of each house

                house.l1_export_rate_no_batt[step] += l1_export_rate
                house.l1_import_rate_no_batt[step] += l1_import_rate

                # Update the energy transacted related information

                if step_offer > 0 and step_demand > 0:
                    
                    step_surplus_distribution = step_transacted_energy * (step_surplus_no_batt / step_offer)
                    step_shortage_distribution = step_transacted_energy * (step_shortage_no_batt / step_demand)

                    house.l1_export_no_batt[step] += step_surplus_distribution[house_ix]
                    house.l1_import_no_batt[step] += step_shortage_distribution[house_ix]
                    house.net_energy_no_batt[step] += (step_surplus_distribution[house_ix] - step_shortage_distribution[house_ix])

    def compute_transactions(self) -> None:

        # Compute the net energy of each house and the surplus/shortage

        step_net_energy = np.stack([house.net_energy[:,self.current_step] for house in self.houses], axis=1)
        step_surplus = np.maximum(0,-step_net_energy)
        step_shortage = np.maximum(0,step_net_energy)

        # Compute the transactions between houses

        for batch_ix in range(self.batch_size):
            
            batch_surplus = step_surplus[batch_ix]
            batch_shortage = step_shortage[batch_ix]
            batch_offer = np.sum(batch_surplus)
            batch_demand = np.sum(batch_shortage)

            # Compute selling and buying rate according to supply and demand

            baseline = self.price[self.current_step] * (1 - self.current_profile['import_fraction'])
            l1_import_rate = baseline + self.current_profile['l1_alpha'] * batch_demand - self.current_profile['l1_beta'] * batch_offer
            l1_export_rate = l1_import_rate - self.current_profile['l1_fee']

            # Update the houses for the current time step

            for house_ix, house in enumerate(self.houses):

                # Update the selling and buying rate of each house

                house.l1_export_rate[batch_ix, self.current_step] = l1_export_rate
                house.l1_import_rate[batch_ix, self.current_step] = l1_import_rate

                # Update the energy transacted related information

                if batch_offer > 0 and batch_demand > 0:

                    # Compute the energy transacted and the contribution of each house

                    batch_transacted_energy = np.minimum(batch_offer, batch_demand)
                    batch_surplus_distribution = batch_transacted_energy * (batch_surplus / batch_offer)
                    batch_shortage_distribution = batch_transacted_energy * (batch_shortage / batch_demand)

                    self.houses[house_ix].l1_export[batch_ix, self.current_step] += batch_surplus_distribution[house_ix]
                    self.houses[house_ix].l1_import[batch_ix, self.current_step] += batch_shortage_distribution[house_ix]
                    self.houses[house_ix].net_energy[batch_ix, self.current_step] += (batch_surplus_distribution[house_ix] - batch_shortage_distribution[house_ix])

    def compute_reward(self) -> np.ndarray:

        # Compute the transactions between houses

        self.compute_transactions()

        # Compute the reward for each house

        rewards = np.stack([house.compute_reward() for house in self.houses], axis=0)

        # Increment the step

        self.increment_step()

        return rewards

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