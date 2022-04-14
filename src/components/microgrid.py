import wandb
import numpy as np

from pandas import DataFrame
from typing import TypedDict
from .pv import PVGeneration, PVParameters, Coordinates, PVCharacteristics
from .load import LoadProfile, LoadTypes, LoadParameters
from .battery import Battery, BatteryParameters
from .generator import Generator, GeneratorParameters
from .grid import Grid, GridParameters


class Architecture(TypedDict):
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
    # charge_battery: bool TODO: Include battery operation


class Microgrid:

    def __init__(self, architecture=None, parameters=None):
        """

        Parameters
        ----------
        architecture : Architecture
            Dict indicating whether a microgrid contains a resource or not.
        parameters : MicrogridParameters
            Dict of parameters for the components of the microgrid.
        """

        # Check empty parameters configuration

        if architecture is None:
            architecture = {
                'pv': True,
                'battery': True,
                'generator': True,
                'grid': True
            }

        if parameters is None:
            parameters = {
                'pv': self.get_default_pv_params(),
                'load': self.get_default_load_params(),
                'battery': self.get_default_battery_params(),
                'generator': self.get_default_generator_params(),
                'grid': self.get_default_grid_params()
            }

        # Initialize the class attributes

        self.architecture = architecture
        self.parameters = parameters
        self._current_t = 0

        # Configure the microgrid

        self._load = LoadProfile(parameters=parameters['load'])
        if architecture['pv']:
            self._pv = PVGeneration(parameters=parameters['pv'])
            self._pv.configure_pv_system()
        if architecture['battery']:
            self._battery = Battery(parameters=parameters['battery'])
        if architecture['generator']:
            self._generator = Generator(parameters=parameters['generator'])
        if architecture['grid']:
            self._grid = Grid(parameters=parameters['grid'])

        # Configure historic data DataFrames

        self._actions_history = DataFrame([])
        self._status_history = DataFrame([])
        self._cost_history = DataFrame([])

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
            use_generator=binary_action[2]
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

    def observe_by_setting_generator(self) -> np.ndarray:

        soc = self._battery.soc
        ghi = self._pv.get_ghi(self.get_current_step())
        pressure = self._pv.get_pressure(self.get_current_step())
        wind_speed = self._pv.get_wind_speed(self.get_current_step())
        air_temperature = self._pv.get_air_temperature(self.get_current_step())
        relative_humidity = self._pv.get_relative_humidity(self.get_current_step())

        return np.array([soc, ghi, pressure, wind_speed, air_temperature, relative_humidity])

    def operation_by_setting_generator(self, power_rate: float) -> (np.ndarray, float):

        load = self._load.get_step_load(self.get_current_step())
        pv = self._pv.get_step_generation(self.get_current_step())
        # grid = 0.0 - TODO: Include grid in this setting
        generator = self._generator.check_generator_constraints(power_rate=power_rate)

        # Decide the interaction with the battery

        remaining_power = generator + pv - load

        p_charge, p_discharge, _ = self._battery.check_battery_constraints(remaining_power=remaining_power)

        # Compute grid operation cost

        cost = (generator - p_discharge) * self._generator.fuel_cost

        wandb.log({
            'load': load,
            'pv': pv,
            'generator': generator,
            'remaining_power': remaining_power,
            'soc': self._battery.soc,
            'cap_to_charge': self._battery.capacity_to_charge,
            'cap_to_discharge': self._battery.capacity_to_discharge,
            'p_charge': p_charge,
            'p_discharge': p_discharge,
            'cost': cost
        })

        # Increase time step

        self._current_t += 1

        return self.observe_by_setting_generator(), cost

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

    @staticmethod
    def get_default_pv_params() -> PVParameters:
        return PVParameters(
            coordinates=Coordinates(
                latitude=24.4274827,
                longitude=54.6234876,
                name='Masdar',
                altitude=0,
                timezone='Asia/Dubai'
            ),
            pv_parameters=PVCharacteristics(
                n_arrays=1,
                modules_per_string=10,
                n_strings=1,
                surface_tilt=20,
                surface_azimuth=180,
                solar_panel_ref='Canadian_Solar_CS5P_220M___2009_',
                inverter_ref='iPower__SHO_5_2__240V_'
            ),
            year=2022
        )

    @staticmethod
    def get_default_load_params():
        return LoadParameters(
            load_type='residential_1'
        )

    @staticmethod
    def get_default_battery_params():
        return BatteryParameters(
            soc=0.1,
            soc_max=0.9,
            soc_min=0.1,
            p_charge_max=0.5,
            p_discharge_max=0.5,
            efficiency=0.9,
            cost_cycle=0.04,
            capacity=4,
            capacity_to_charge=3.2,
            capacity_to_discharge=0
        )

    @staticmethod
    def get_default_generator_params():
        return GeneratorParameters(
            rated_power=2.5,
            p_max=0.9,
            p_min=0.1,
            fuel_cost=0.4,
            co2=2
        )

    @staticmethod
    def get_default_grid_params():
        return GridParameters(
            max_export=50,
            max_import=50,
            price_export=0.01,
            price_import=0.08,
            status=True,
            co2=2
        )
