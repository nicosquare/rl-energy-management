import pvlib
import pandas as pd
import matplotlib.pyplot as plt

from pvlib.forecast import GFS
from pvlib.pvsystem import PVSystem, FixedMount, Array
from pvlib.location import Location
from pvlib.modelchain import ModelChain


class PVGeneration:

    def __init__(self, coordinates):
        self._model_chain = None
        self.coordinates = coordinates

    def configure_pv_system(
            self, n_arrays=1, modules_per_string=10, n_strings=1, surface_tilt=20, surface_azimuth=180,
            solar_panel_ref='Canadian_Solar_CS5P_220M___2009_',
            inverter_ref='iPower__SHO_5_2__240V_'
    ):
        """

        Set up the solar panel configuration to be used.

        Parameters
        ----------
        n_arrays: int
            Number of solar panel arrays
        modules_per_string: int
            Number of solar panels per string
        n_strings: int
            Number of strings per array
        surface_tilt: int
            Degrees of tilt of each module mount.
        surface_azimuth: int
            Azimuth of the module surface over the mount.
        solar_panel_ref: string
            Reference of the solar panel module as it appear in SAM
        inverter_ref: string
            Reference of the solar panel inverter as it appear in SAM

        Returns
        -------

        """
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

    def _get_tmy(self, year):
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
            f'{year}-01-01T00:00:00', f'{year}-12-31T23:59:59', freq='60T', tz=timezone
        )
        weather.index.name = "utc_time"

        return weather

    def get_estimate_generation(self, year):
        """
        Returns an estimation of the ac power, with 1h resolution, for a year of the defined solar panel
        Parameters
        ----------
        year: int
            Year to be estimated

        Returns
        -------
        ac: DataFrame
            Hourly AC power for the given year with the configured solar panel
        """

        weather = self._get_tmy(year=year)
        self._model_chain.run_model(weather)

        return self._model_chain.results.ac

    def get_forecast_generation(self, start, n_days_ahead=7):
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

    def get_ghi(self, year):
        return self._get_tmy(year=year)['ghi']

    def get_gni(self, year):
        return self._get_tmy(year=year)['dni']

    def get_dhi(self, year):
        return self._get_tmy(year=year)['dhi']

    def get_air_temperature(self, year):
        return self._get_tmy(year=year)['temp_air']

    def get_wind_speed(self, year):
        return self._get_tmy(year=year)['wind_speed']

    def get_relative_humidity(self, year):
        return self._get_tmy(year=year)['relative_humidity']

    def get_pressure(self, year):
        return self._get_tmy(year=year)['pressure']