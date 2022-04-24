import torch
from torch import Tensor
from typing import TypedDict


class GeneratorParameters(TypedDict):
    """
        rated_power: float
            Maximum rated power of the generator (kW).
        p_min: float
            Value representing the minimum operating power of the generator (%)
        p_max: float
            Value representing the maximum operating power of the generator (%)
        fuel_cost: float
            Value representing the cost of using the generator ($/kWh).
        co2: float
            Carbon footprint of the energy using this generator (CO2 g./kWh)
    """
    rated_power: float
    p_min: float
    p_max: float
    fuel_cost: float
    co2: float


class Generator:
    def __init__(self, parameters=None):
        """

        Class that contains the characteristics of a local source of energy that uses fossil fuels.

        Parameters
        ----------
        parameters : GeneratorParameters

            Dict of configurations with the following shape:

                {
                    rated_power: int,
                    p_min: float,
                    p_max: float,
                    fuel_cost: float,
                    co2: float
                }

        """

        # Check empty parameters configuration

        if parameters is None:
            parameters = {
                'rated_power': 1.5,
                'p_min': 0.9,
                'p_max': 0.1,
                'fuel_cost': 0.5,
                'co2': 2.0
            }

        # Initialize the class attributes

        self.fuel_cost = parameters['fuel_cost']
        self.p_max = parameters['p_max']
        self.p_min = parameters['p_min']
        self.rated_power = parameters['rated_power']
        self.co2 = parameters['co2']

    def check_constraints(self, input_rate: Tensor):
        """
            Check the generator limits to define the output power.
        Parameters
        ----------
        input_rate: Tensor
            Demanded generator power (kW).

        Returns
        -------
            output_power: Tensor
                Generator output power according to the constraints check.
        """

        # Check upper limit

        real_rate = torch.where(input_rate > self.p_max, self.p_max, input_rate)

        # Check lower limit

        real_rate = torch.where(real_rate < self.p_min, self.p_min, real_rate)

        return real_rate * self.rated_power
