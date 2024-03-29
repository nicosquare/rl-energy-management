import numpy as np

from typing import Tuple, TypedDict

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
        buy_price: float
            Price for using energy from the battery ($/kWh).
        sell_price: float
            Price for injecting energy to the battery (reward to the prosumers).
    """
    capacity: float
    soc_max: float
    soc_min: float
    p_charge_max: float
    p_discharge_max: float
    efficiency: float
    buy_price: float
    sell_price: float


class Battery:
    def __init__(self, params=None, batch_size: int = 1, random_soc_0: bool = False):
        """

        Representation of a Battery and the basic parameters that define its operation.

        Parameters
        ----------
            batch_size : int
                Number of batteries to represent in the case of a batch training/testing.
            params: BatteryParameters

                Dict of configurations with the following shape:

                {
                    capacity: float,
                    soc_max: float,
                    soc_min: float,
                    p_charge_max: float,
                    p_discharge_max: float,
                    efficiency: float,
                    buy_price: float,
                    sell_price: float,
                }

        """

        # Check empty parameters configuration

        if params is None:
            params = {
                'capacity': 1.0,
                'soc_max': 0.9,
                'soc_min': 0.1,
                'p_charge_max': 0.5,
                'p_discharge_max': 0.5,
                'efficiency': 0.9,
                'buy_price': 0.6,
                'sell_price': 0.6
            }

        # Initialize the class attributes

        self.batch_size = batch_size
        self.capacity = params['capacity']
        self.soc_max = params['soc_max']
        self.soc_min = params['soc_min']
        self.p_charge_max = params['p_charge_max']
        self.p_discharge_max = params['p_discharge_max']
        self.efficiency = params['efficiency']
        self.buy_price = params['buy_price']
        self.sell_price = params['sell_price']
        self.capacity_to_charge = None
        self.capacity_to_discharge = None
        self.random_soc_0 = random_soc_0
        self.soc = self.initialize_soc(is_random=random_soc_0)

        # History of the SoC, Power of charge and Power of discharge

        self.hist_soc = np.copy(self.soc)
        self.hist_p_charge = np.zeros((batch_size, 1))
        self.hist_p_discharge = np.zeros((batch_size, 1))

        # Initialize the capacity status

        self.compute_capacity_status()

    def initialize_soc(self, is_random: bool = False):
        """
            Initialize the SoC according to the batch size
        :return:

            init_values: Array:
                Array with the initialization values.
        """
        return np.ones((self.batch_size, 1)) * self.soc_min if not is_random else np.random.uniform(low=self.soc_min, high=self.soc_max, size=(self.batch_size, 1))

    def reset(self):
        """
            Reset the battery to the initialization state.
        :return:
            None
        """
        self.soc = self.initialize_soc(is_random=self.random_soc_0)
        self.hist_soc = np.copy(self.soc)
        self.hist_p_charge = np.zeros((self.batch_size, 1))
        self.hist_p_discharge = np.zeros((self.batch_size, 1))
        self.capacity_to_charge = None
        self.capacity_to_discharge = None

    def compute_capacity_status(self):
        """
           Computes the capacity need to fully charge and completely discharge the battery.
        Returns
        -------
            None
        """
        self.capacity_to_charge = np.maximum(
            (self.soc_max * self.capacity - self.soc * self.capacity) / self.efficiency,
            np.zeros((self.batch_size, 1))
        )

        self.capacity_to_discharge = np.maximum(
            (self.soc * self.capacity - self.soc_min * self.capacity) / self.efficiency,
            np.zeros((self.batch_size, 1))
        )

    def check_battery_constraints(self, power_rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            Check the physical constrains of the battery to define the maximum power it can charge or discharge.
        Parameters
        ----------
        input_power: Array
            Amount of energy that is required from the battery, could be positive for charging or negative for
            discharging.

        Returns
        -------
            p_charge: Array
                Power of charge given the input power.
            p_discharge: Array
                Power of discharge given the input power.
            new_soc: Array
                Value between 0 and 1 indicating the new SoC after charging or discharging.
        """

        # Initialize the charge and discharge power

        input_power = power_rate * self.capacity
        p_charge = np.maximum(input_power, np.zeros((self.batch_size, 1)))
        p_discharge = np.maximum(-input_power, np.zeros((self.batch_size, 1)))

        # Compute the capacities to charge or discharge

        self.compute_capacity_status()

        # Check battery constraints

        min_charge = np.minimum(
            self.capacity_to_charge,
            np.ones((self.batch_size, 1)) * self.p_charge_max * self.capacity
        )

        p_charge = np.where(
            p_charge > min_charge,
            min_charge,
            p_charge
        )

        max_discharge = np.minimum(
            self.capacity_to_discharge,
            np.ones((self.batch_size, 1)) * self.p_discharge_max * self.capacity
        )

        p_discharge = np.where(
            p_discharge > max_discharge,
            max_discharge,
            p_discharge
        )

        # Compute the ineffective action power (tried to charge or discharge more than the battery can)

        i_action = np.where(
            power_rate > 0,
            np.abs(p_charge - input_power),
            np.abs(p_discharge + input_power)
        ).squeeze()

        return p_charge, p_discharge, i_action

    def apply_action(self, p_charge: np.ndarray, p_discharge: np.ndarray):
        """
            Compute the new SoC according to an instruction for charging/discharging.
        :param p_charge: Array indicating the power of charge for the batch battery.
        :param p_discharge: Array indicating the power of discharge for the batch battery.
        :return:
            None
        """
        
        self.soc = self.soc + (p_charge * self.efficiency - p_discharge / self.efficiency) / self.capacity
        self.soc = self.soc.clip(min=self.soc_min, max=self.soc_max)

        # Update the history of the SoC, Power of charge and Power of discharge
        
        self.hist_soc = np.append(self.hist_soc, self.soc, axis=1)
        self.hist_p_charge = np.append(self.hist_p_charge, p_charge, axis=1)
        self.hist_p_discharge = np.append(self.hist_p_discharge, p_discharge, axis=1)

        # Compute the capacities to charge or discharge

        self.compute_capacity_status()