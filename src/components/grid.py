import torch

from torch import Tensor
from typing import TypedDict


class GridParameters(TypedDict):
    """
        max_export: float
            Value representing the maximum export power to the grid (kW).
        max_import: float
            Value representing the maximum import power from the grid (kW).
        price_export: float
            Value representing the cost of exporting to the grid (currency/kWh).
        price_import: float
            Value representing the cost of importing to the grid (currency/kWh).
        status: int, boolean
            Binary value representing whether the grid is connected or not (for example 0 represent a black-out of the
            main grid).
        co2: float
            Carbon footprint of the energy using this generator (CO2 g./kWh).
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
                    status: bool,
                    co2: float
                }

        """

        # Check empty parameters configuration

        if parameters is None:
            parameters = {
                'max_export': 50.0,
                'max_import': 50.0,
                'price_export': 0.25,
                'price_import': 0.8,
                'status': True,
                'co2': 1.0
            }

        # Initialize the class attributes

        self.max_export = parameters['max_export']
        self.max_import = parameters['max_import']
        self.price_export = parameters['price_export']
        self.price_import = parameters['price_import']
        self.status = parameters['status']
        self.co2 = parameters['co2']

    def check_constraints(self, input_power: Tensor):
        """
            Check the generator limits to define the output power.
        Parameters
        ----------
        input_power: Tensor
            Demanded grid power (kW).

        Returns
        -------
            output_power: Tensor
                Grid output power according to the constraints check.
        """

        # Check upper limit

        real_power = torch.where(input_power > self.max_import, self.max_import, input_power)

        # Check lower limit

        real_power = torch.where(-real_power > self.max_export, self.max_export, real_power)

        return real_power
