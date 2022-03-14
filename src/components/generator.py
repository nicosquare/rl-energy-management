from typing import TypedDict


class GeneratorParameters(TypedDict):
    """
        rated_power: int
            Maximum rated power of the generator.
        p_min: float
            Value representing the minimum operating power of the generator (kW)
        p_max: float
            Value representing the maximum operating power of the generator (kW)
        fuel_cost: float
            Value representing the cost of using the generator in $/kWh.
        co2: float
            Carbon footprint of the energy using this generator in grams of CO2/kWh
    """
    rated_power: int
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
                'rated_power': 0,
                'p_min': 0.0,
                'p_max': 0.0,
                'fuel_cost': 0.0,
                'co2': 0.0
            }

        # Initialize the class attributes

        self.fuel_cost = parameters['fuel_cost']
        self.p_max = parameters['p_max']
        self.p_min = parameters['p_min']
        self.rated_power = parameters['rated_power']
        self.co2 = parameters['co2']
