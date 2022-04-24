import torch

from torch import Tensor
from typing import TypedDict


class BatteryParameters(TypedDict):
    """
        soc_0: float
            Value between 0 and 1 representing the initial state of charge of the battery (1 being full, 0 being empty).
        capacity: int
            Energy capacity of the battery/array of batteries in (kWh).
        soc_max: float
            Value between 0 and 1 representing the highest value the SoC a battery can reach.
        soc_min: float
            Value between 0 and 1 representing the lowest value the SoC a battery can reach.
        p_charge_max: float
            Maximum charging rate of a battery (%).
        p_discharge_max: float
            Minimum charging rate of a battery (%).
        efficiency: float
            Value between 0 and 1 representing a one-way efficiency of the battery considering same efficiency for
            charging and discharging (%).
        cost_cycle: float
            Cost of completing an energy charging/discharging cycle (currency/kWh).
    """
    soc_0: float
    capacity: int
    soc_max: float
    soc_min: float
    p_charge_max: float
    p_discharge_max: float
    efficiency: float
    cost_cycle: float


class Battery:
    def __init__(self, params=None, batch_size: int = 1):
        """

        Representation of a Battery and the basic parameters that define its operation.

        Parameters
        ----------
            batch_size : int
                Number of batteries to represent in the case of a batch training/testing.
            params: BatteryParameters

                Dict of configurations with the following shape:

                {
                    soc_0: float,
                    capacity: int,
                    soc_max: float,
                    soc_min: float,
                    p_charge_max: float,
                    p_discharge_max: float,
                    efficiency: float,
                    cost_cycle: float
                }

        """

        # Check empty parameters configuration

        if params is None:
            params = {
                'soc_0': 0.0,
                'capacity': 1.0,
                'soc_max': 0.9,
                'soc_min': 0.1,
                'p_charge_max': 0.5,
                'p_discharge_max': 0.5,
                'efficiency': 0.9,
                'cost_cycle': 1.0
            }

        # Initialize the class attributes

        self.batch_size = batch_size
        self.soc = torch.full((batch_size,), params['soc_0'])
        self.capacity = params['capacity']
        self.soc_max = params['soc_max']
        self.soc_min = params['soc_min']
        self.p_charge_max = params['p_charge_max']
        self.p_discharge_max = params['p_discharge_max']
        self.efficiency = params['efficiency']
        self.cost_cycle = params['cost_cycle']
        self.capacity_to_charge = None
        self.capacity_to_discharge = None

        # Initialize the capacity status

        self.compute_capacity_status()

    def compute_capacity_status(self):
        """
           Computes the capacity need to fully charge and completely discharge the battery.
        Returns
        -------
            None
        """
        self.capacity_to_charge = torch.maximum(
            (self.soc_max * self.capacity - self.soc * self.capacity) / self.efficiency,
            torch.zeros(self.batch_size)
        )

        self.capacity_to_discharge = torch.maximum(
            (self.soc * self.capacity - self.soc_min * self.capacity) / self.efficiency,
            torch.zeros(self.batch_size)
        )

    def check_battery_constraints(self, input_power: Tensor):
        """
            Check the physical constrains of the battery to define the maximum power it can charge or discharge.
        Parameters
        ----------
        input_power: Tensor
            Amount of energy that is required from the battery, could be positive for charging or negative for
            discharging.

        Returns
        -------
            p_charge: Tensor
                Power of charge given the input power.
            p_discharge: Tensor
                Power of discharge given the input power.
            new_soc: Tensor
                Value between 0 and 1 indicating the new SoC after charging or discharging.
        """

        # Initialize the charge and discharge power

        p_charge = torch.maximum(input_power, torch.zeros(self.batch_size))
        p_discharge = torch.maximum(-input_power, torch.zeros(self.batch_size))

        # Compute the capacities to charge or discharge

        self.compute_capacity_status()

        # Check battery constraints

        min_charge = torch.minimum(self.capacity_to_charge, torch.ones(self.batch_size) * self.p_charge_max)

        p_charge = torch.where(
            p_charge > min_charge,
            min_charge,
            p_charge
        )

        max_discharge = torch.minimum(self.capacity_to_discharge, torch.ones(self.batch_size) * self.p_discharge_max)

        p_discharge = torch.where(
            p_discharge > max_discharge,
            max_discharge,
            p_discharge
        )

        # Compute new SoC

        self.soc = self.soc + (p_charge*self.efficiency - p_discharge/self.efficiency) / self.capacity

        return p_charge, p_discharge, self.soc
