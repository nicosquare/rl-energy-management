from typing import TypedDict


class GridParameters(TypedDict):
    """
        max_export: float
            Value representing the maximum export power to the grid (kW).
        max_import: float
            Value representing the maximum import power from the grid (kW).
        price_export: float
            Value representing the cost of exporting to the grid in currency/kWh.
        price_import: float
            Value representing the cost of importing to the grid in currency/kWh.
        status: int, boolean
            Binary value representing whether the grid is connected or not (for example 0 represent a black-out of the
            main grid).
        co2: float
            Carbon footprint of the energy using this generator in grams of CO2/kWh
    """
    max_export: float
    max_import: float
    price_export: float
    price_import: float
    status: bool
    co2: float


class Grid:

    def __init__(self, parameters=None):
        """

        Class that defines the operation conditions for the energy grid.

        Parameters
        ----------
        parameters : GridParameters

            Dict of configurations with the following shape:

                {
                    max_export: float,
                    max_import: float,
                    price_export: float,
                    price_import: float,
                    status: float,
                    co2: float
                }

        """

        # Check empty parameters configuration

        if parameters is None:
            parameters = {
                'max_export': 0.0,
                'max_import': 0.0,
                'price_export': 0.0,
                'price_import': 0.0,
                'status': 0.0,
                'co2': 0.0
            }

        # Initialize the class attributes

        self.max_export = parameters['max_export']
        self.max_import = parameters['max_import']
        self.price_export = parameters['price_export']
        self.price_import = parameters['price_import']
        self.status = parameters['status']
        self.co2 = parameters['co2']
