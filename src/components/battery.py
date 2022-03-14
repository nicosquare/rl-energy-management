from typing import TypedDict


class BatteryParameters(TypedDict):
    """
        soc: float
            Value between 0 and 1 representing the state of charge of the battery (1 being full, 0 being empty). The
            initialization value is assumed as the initial SoC.
        capacity: int
            Energy capacity of the battery/array of batteries in (kWh).
        soc_max: float
            The highest value the SoC a battery can reach.
        soc_min: float
            The lowest value the SoC a battery can reach.
        p_charge_max: float
            Maximum charging rate of a battery (kW).
        p_discharge_max: float
            Minimum charging rate of a battery (kW).
        efficiency: float
            Value between 0 and 1 representing a one-way efficiency of the battery (considering same efficiency for
            charging and discharging).
        cost_cycle: float
            Cost of using the battery in currency/kWh.
        capacity_to_charge: float
            Represents the amount of energy that a battery can charge before being full.
        capacity_to_discharge: float
            Represents the amount of energy available that a battery can discharge before being empty.
    """
    soc: float
    capacity: int
    soc_max: float
    soc_min: float
    p_charge_max: float
    p_discharge_max: float
    efficiency: float
    cost_cycle: float
    capacity_to_charge: float
    capacity_to_discharge: float


class Battery:

    def __init__(self, parameters=None):
        """

        Representation of a Battery and the basic parameters that define its operation.

        Parameters
        ----------
            parameters: BatteryParameters

                Dict of configurations with the following shape:

                {
                    soc: float,
                    capacity: int,
                    soc_max: float,
                    soc_min: float,
                    p_charge_max: float,
                    p_discharge_max: float,
                    efficiency: float,
                    cost_cycle: float,
                    capacity_to_charge: float,
                    capacity_to_discharge: float
                }

        """

        # Check empty parameters configuration

        if parameters is None:
            parameters = {
                'soc': 0.0,
                'capacity': 0.0,
                'soc_max': 0.0,
                'soc_min': 0.0,
                'p_charge_max': 0.0,
                'p_discharge_max': 0.0,
                'efficiency': 0.0,
                'cost_cycle': 0.0,
                'capacity_to_charge': 0.0,
                'capacity_to_discharge': 0.0,
            }

        # Initialize the class attributes

        self.soc = parameters['soc']
        self.capacity = parameters['capacity']
        self.soc_max = parameters['soc_max']
        self.soc_min = parameters['soc_min']
        self.p_charge_max = parameters['p_charge_max']
        self.p_discharge_max = parameters['p_discharge_max']
        self.efficiency = parameters['efficiency']
        self.cost_cycle = parameters['cost_cycle']
        self.capacity_to_charge = parameters['capacity_to_charge']
        self.capacity_to_discharge = parameters['capacity_to_discharge']
