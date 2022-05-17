import torch
import numpy as np

from typing import TypedDict
from torch import Tensor

from .pv import PVGeneration, PVParameters
from .load import LoadProfile, LoadParameters
from .battery import Battery, BatteryParameters
from .generator import Generator, GeneratorParameters
from .grid import Grid, GridParameters
from ..utils.tensors import create_ones_tensor, create_zeros_tensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)


class MicrogridArchitecture(TypedDict):
    pv: bool
    battery: bool
    generator: bool
    grid: bool


class MicrogridParameters(TypedDict):
    pv: PVParameters
    load: LoadParameters
    battery: BatteryParameters
    generator: GeneratorParameters
    grid: GridParameters


class MicrogridAction(TypedDict):
    use_pv: bool
    use_generator: bool
    use_grid: bool
    charge_battery: bool


class Microgrid:

    def __init__(self, batch_size: int, arch: MicrogridArchitecture = None, params: MicrogridParameters = None):
        """

        Parameters
        ----------
        batch_size: int
            Number of parallel runs of the current microgrid.
        arch : MicrogridArchitecture
            Dict indicating whether a microgrid contains a resource or not.
        params : MicrogridParameters
            Dict of parameters for the components of the microgrid.
        """

        # Check empty parameters configuration

        if arch is None:
            arch = {
                'pv': True,
                'battery': True,
                'generator': True,
                'grid': True
            }

        # Initialize the class attributes

        self.batch_size = batch_size
        self.architecture = arch
        self._current_t = 0

        # Configure the microgrid

        self._load = LoadProfile(params=params['load'] if params is not None else None)
        if arch['pv']:
            self._pv = PVGeneration(params=params['pv'] if params is not None else None)
            self._pv.configure_pv_system()
        if arch['battery']:
            self._battery = Battery(batch_size=batch_size, params=params['battery'] if params is not None else None)
        if arch['generator']:
            self._generator = Generator(parameters=params['generator'] if params is not None else None)
        if arch['grid']:
            self._grid = Grid(parameters=params['grid'] if params is not None else None)

    def observe_by_source_selection(self) -> np.ndarray:
        load_t = self._load.get_step_load(self.get_current_step())
        pv_t = 0

        if self.architecture['pv']:
            pv_t = self._pv.get_step_generation(self.get_current_step())

        return np.array([load_t, pv_t])

    def operation_by_source_selection(self, action: int) -> (float, float, float):

        # Process the action as MicrogridAction

        binary_action = [bit == '1' for bit in "{0:3b}".format(0)]

        action = MicrogridAction(
            use_pv=binary_action[0],
            use_grid=binary_action[1],
            use_generator=binary_action[2],
            charge_battery=False
        )

        load, pv = self.observe_by_source_selection()
        grid = 0.0
        generator = 0.0
        available_grid_supply = 0.0
        # battery = 0.0

        # Surplus could be negative (there is lack of energy) or positive (energy to export).

        surplus = pv - load

        # Is there is a lack of energy we get energy from the grid, when there is excess the energy we export.

        if self.architecture['grid'] and action['use_grid']:

            available_grid_supply = self._grid.max_export + surplus

            # If available grid supply is negative, the grid cannot meet the load, we need extra generation
            if available_grid_supply < 0:
                grid = self._grid.max_export
            else:
                # A positive grid value means its exporting energy, a negative that its importing
                grid = -surplus

        # In the case the grid is not able to provide energy, we get the energy from the generator

        generator_max_power = self._generator.p_max * self._generator.rated_power

        if self.architecture['generator'] and action['use_generator'] and action['use_grid']:

            available_generator_supply = generator_max_power + available_grid_supply

            # If available generator supply is negative, there is not enough energy in the microgrid
            if available_generator_supply < 0:
                generator = generator_max_power
            # Lack of grid energy, the generator gives the extra energy.
            elif available_generator_supply < generator_max_power:
                generator = available_grid_supply

        # If the grid is not enabled, we supply the rest of energy with the generator

        if self.architecture['generator'] and action['use_generator'] and not action['use_grid']:

            available_generator_supply = generator_max_power + surplus

            # If available generator supply is negative, there is not enough energy in the microgrid
            if available_generator_supply < 0:
                generator = generator_max_power
            else:
                # If there is surplus, it is just discarded as there is no grid.
                generator = max(-surplus, 0)

        # Check and compute if there is unmet load

        unmet_load = load - pv - grid - generator

        # Compute grid operation cost

        cost = grid * self._grid.price_export if grid > 0 else grid * self._grid.price_import
        cost += generator * self._generator.fuel_cost
        cost += unmet_load * 1.5 * self._grid.price_export if unmet_load > 0 else 0

        # Increase time step

        self._current_t += 1

        return self.observe_by_source_selection(), cost

    def observe_by_setting_generator(self) -> Tensor:

        # Get the observations

        soc = self._battery.soc
        ghi = self._pv.get_ghi(self.get_current_step())
        pressure = self._pv.get_pressure(self.get_current_step())
        wind_speed = self._pv.get_wind_speed(self.get_current_step())
        air_temperature = self._pv.get_air_temperature(self.get_current_step())
        relative_humidity = self._pv.get_relative_humidity(self.get_current_step())

        # The only observation that could change depending on the action is the SoC

        return torch.stack((
            soc,
            create_ones_tensor(len(soc)) * ghi,
            create_ones_tensor(len(soc)) * pressure,
            create_ones_tensor(len(soc)) * wind_speed,
            create_ones_tensor(len(soc)) * air_temperature,
            create_ones_tensor(len(soc)) * relative_humidity,
        ), dim=1)

    def operation_by_setting_generator(self, power_rate: Tensor) -> (np.ndarray, float):

        load = self._load.get_step_load(self.get_current_step())
        pv = self._pv.get_step_generation(self.get_current_step())
        generator = self._generator.check_constraints(input_rate=power_rate)

        # Decide the interaction with the battery

        remaining_power = generator + pv - load

        p_charge, p_discharge = self._battery.check_battery_constraints(input_power=remaining_power)
        self._battery.compute_new_soc(p_charge=p_charge, p_discharge=p_discharge)

        # Check if all the energy is attended

        power_after_battery = remaining_power + p_discharge
        unattended_power = torch.maximum(-power_after_battery, create_zeros_tensor(self.batch_size))

        # Compute grid operation cost, unattended power is penalized with more expensive fuel

        cost = (generator - p_discharge + unattended_power * 1.5) * self._generator.fuel_cost

        # Compute next state

        next_state = self.observe_by_setting_generator()

        # Increase time step

        self._current_t += 1

        return next_state, cost

    def get_current_step(self):
        """
            Returns the current time step.
        Returns
        -------
            self.current_t: int
                Current microgrid time step
        """
        return self._current_t % 8760

    def reset_current_step(self):
        """
            Resets the current time step.
        Returns
        -------
            None
        """
        self._current_t = 0
