import pvlib
import pandas as pd

from typing import TypedDict, Union
from pvlib.forecast import GFS
from pvlib.pvsystem import PVSystem, FixedMount, Array
from pvlib.location import Location
from pvlib.modelchain import ModelChain


class Coordinates(TypedDict):
    latitude: float
    longitude: float
    name: str
    altitude: float
    timezone: str


class PVCharacteristics(TypedDict):
    """
        n_arrays: int
            Number of solar panel arrays.
        modules_per_string: int
            Number of solar panels per string.
        n_strings: int
            Number of strings per array.
        surface_tilt: int
            Degrees of tilt of each module mount.
        surface_azimuth: int
            Azimuth of the module surface over the mount.
        solar_panel_ref: string
            Reference of the solar panel module as it appear in SAM.
        inverter_ref: string
            Reference of the solar panel inverter as it appear in SAM.
    """
    n_arrays: int
    modules_per_string: int
    n_strings: int
    surface_tilt: int
    surface_azimuth: int
    solar_panel_ref: str
    inverter_ref: str


class PVParameters(TypedDict):
    """
        coordinates: Coordinates
            Dictionary with the location info of the PV system.
        year: int
            Year for the estimation.
    """
    coordinates: Coordinates
    pv_parameters: PVCharacteristics
    year: int


class PVGeneration:

    def __init__(self, params=None):
        """

        This represents a PV generation system located in a particular location in a defined year.

        Parameters
        ----------
        params: PVParameters

            Dict of configurations with the following shape:

            {
                coordinates: Coordinates,
                year: int
            }

        """
        # Check empty parameters configuration

        if params is None:
            params = PVParameters(
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

        # Initialize the class attributes

        self.coordinates = params['coordinates']
        self.pv_parameters = params['pv_parameters']
        self.year = params['year']
        self._model_chain = None
        self._weather_ts = None
        self._generation_ts = None

    def configure_pv_system(self):
        """

        Set up the solar panel configuration to be used.

        Returns
        -------
            None
        """

        n_arrays = self.pv_parameters['n_arrays']
        modules_per_string = self.pv_parameters['modules_per_string']
        n_strings = self.pv_parameters['n_strings']
        surface_tilt = self.pv_parameters['surface_tilt']
        surface_azimuth = self.pv_parameters['surface_azimuth']
        solar_panel_ref = self.pv_parameters['solar_panel_ref']
        inverter_ref = self.pv_parameters['inverter_ref']

        # Configure the solar panel and inverter specifications from SAM (Default)

        sandia_modules = pvlib.pvsystem.retrieve_sam('sandiamod')
        sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

        module_params = sandia_modules[solar_panel_ref]
        inverter_params = sapm_inverters[inverter_ref]
        temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

        mount = FixedMount(surface_tilt=surface_tilt, surface_azimuth=surface_azimuth)

        location = Location(
            latitude=self.coordinates['latitude'],
            longitude=self.coordinates['longitude'],
            name=self.coordinates['name'],
            altitude=self.coordinates['altitude'],
            tz=self.coordinates['timezone'],
        )

        # Build PV array

        pv_array = []

        for _ in range(n_arrays):
            pv_array.append(
                Array(
                    mount=mount,
                    module_parameters=module_params,
                    temperature_model_parameters=temp_params,
                    modules_per_string=modules_per_string,
                    strings=n_strings
                )
            )

        # Define the system for the instance

        pv_system = PVSystem(arrays=pv_array, inverter_parameters=inverter_params)

        self._model_chain = ModelChain(system=pv_system, location=location, aoi_model='no_loss',
                                       spectral_model='no_loss')

        # Get the generation estimation and weather initialization

        self._weather_ts = self._get_tmy()
        self._generation_ts = self._get_estimate_generation()

    def _get_tmy(self):
        """

        Returns a typical meteorological year with resolution of 1 hour (8760 data points) for the PV location.

        Returns
        -------
            weather: DataFrame
                Contains a set with the following meteorological attributes:
                    temp_air           float64
                    relative_humidity  float64
                    ghi                float64
                    dni                float64
                    dhi                float64
                    IR(h)              float64
                    wind_speed         float64
                    wind_direction     float64
                    pressure
        """
        latitude = self.coordinates['latitude']
        longitude = self.coordinates['longitude']
        timezone = self.coordinates['timezone']

        weather = pvlib.iotools.get_pvgis_tmy(latitude, longitude, map_variables=True)[0]
        weather.index = pd.date_range(
            f'{self.year}-01-01T00:00:00', f'{self.year}-12-31T23:59:59', freq='60T', tz=timezone
        )
        weather.index.name = "utc_time"

        return weather

    def _get_estimate_generation(self):
        """
        Returns an estimation of the ac power, with 1h resolution, for a year of the defined solar panel.
        Parameters
        ----------

        Returns
        -------
        ac: DataFrame
            Hourly AC power for the given year with the configured solar panel.
        """

        weather = self._get_tmy()
        self._model_chain.run_model(weather)

        return self._model_chain.results.ac

    def get_step_generation(self, time_step: int = 0):
        """

        Get generation in kW at current time step, also increase the current time step.

        Parameters
        ----------
            time_step : int
                Value of the required time step to get the value from.
        Returns
        -------
            pv_power_t: float

                Output power of the PV system at current time step.

        """
        pv_power_t = self._generation_ts[time_step]
        return pv_power_t / 1000  # Convert to kW

    def get_forecast_generation(self, start: Union[int, str], n_days_ahead: int = 7):
        """

        Forecast generation n_days_ahead from a starting date (that should be at max. 30 before current day).

        Parameters
        ----------
        start: date

            String with the ISO format date string.

        n_days_ahead: int

            Days ahead to consider in the prediction.

        Returns
        -------

        """

        latitude = self.coordinates['latitude']
        longitude = self.coordinates['longitude']
        timezone = self.coordinates['timezone']

        initial_date = pd.Timestamp(start, tz=timezone)
        final_date = initial_date + pd.Timedelta(days=n_days_ahead)

        # GFS model, defaults to 0.5 degree resolution, 0.25 deg available

        fx_model = GFS()
        fx_data = fx_model.get_processed_data(latitude, longitude, initial_date, final_date)
        fx_data = fx_data.resample('1h').interpolate()

        self._model_chain.run_model(fx_data)

        return self._model_chain.results.ac

    def get_ghi(self, time_step: int = 0):
        """

        Get Global Horizontal Irradiation at time current time step.
        
            time_step : int
                Value of the required time step to get the value from.
        Returns
        -------
            ghi: float
                Global Horizontal Irradiation.
        """
        return self._weather_ts['ghi'][time_step]

    def get_dni(self, time_step: int = 0):
        """

        Get Direct Normal Irradiation at time current time step.

            time_step : int
                Value of the required time step to get the value from.

        Returns
        -------
            dni: float
                Direct Normal Irradiation.
        """
        return self._weather_ts['dni'][time_step]

    def get_dhi(self, time_step: int = 0):
        """

        Get Diffused Horizontal Irradiation at time current time step.

            time_step : int
                Value of the required time step to get the value from.

        Returns
        -------
            dhi: float
                Diffused Horizontal Irradiation.
        """
        return self._weather_ts['dhi'][time_step]

    def get_air_temperature(self, time_step: int = 0):
        """

        Get Air Temperature at time current time step.

            time_step : int
                Value of the required time step to get the value from.

        Returns
        -------
            air_temperature: float
                Air Temperature.
        """
        return self._weather_ts['temp_air'][time_step]

    def get_wind_speed(self, time_step: int = 0):
        """

        Get Wind Speed at time current time step.

            time_step : int
                Value of the required time step to get the value from.

        Returns
        -------
            wind_speed: float
                Wind Speed.
        """
        return self._weather_ts['wind_speed'][time_step]

    def get_relative_humidity(self, time_step: int = 0):
        """

        Get Relative Humidity at time current time step.

            time_step : int
                Value of the required time step to get the value from.

        Returns
        -------
            relative_humidity: float
                Relative Humidity.
        """
        return self._weather_ts['relative_humidity'][time_step]

    def get_pressure(self, time_step: int = 0):
        """

        Get Pressure at time current time step.

            time_step : int
                Value of the required time step to get the value from.

        Returns
        -------
            pressure: float
                Pressure.
        """
        return self._weather_ts['pressure'][time_step]
