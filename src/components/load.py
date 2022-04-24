import pandas as pd

from typing import TypedDict, Literal


LoadTypes = Literal[
    'industrial_1', 'industrial_2', 'industrial_3', 'public_1', 'public_2', 'residential_1',
    'residential_2', 'residential_3', 'residential_4', 'residential_5', 'residential_6'
]


class LoadParameters(TypedDict):
    """
        load_type: LoadTypes
            String that describes the kind of load according to the possible values in LoadTypes.
    """
    load_type: LoadTypes


class LoadProfile:

    def __init__(self, params=None):
        """

        Represents the load of an energy user of a defined type in kW

        Parameters
        ----------
        params : LoadParameters

            Dict of configurations with the following shape:

            {
                load_type: LoadTypes
            }

        """

        # Check empty parameters configuration

        if params is None:
            params = LoadParameters(load_type='residential_1')

        # Initialize the class attributes

        self.load_type = params['load_type']
        self._load_ts = self._get_year_data()

    def _get_year_data(self):
        """

        Get from the sample dataset the load of a year

        Returns
        -------

            load_data: pd.DataFrame

                DataFrame containing the year of the selected type of load.

        """
        load_data = pd.read_csv('src/resources/opsd_household_data.csv')

        return load_data[f'{self.load_type}_grid_import'].diff(-1).abs().fillna(0)

    def get_step_load(self, time_step: int = 0):
        """

        Get the load at the current time step, also increase the time step.

            time_step : int
                Value of the required time step to get the value from.

        Returns
        -------

            load_t: float

                Load at current time step in kW.

        """
        load_t = self._load_ts[time_step]

        return load_t

    # TODO: Define a function that predicts coming load from previous n-steps
