from pv import PVGeneration

if __name__ == '__main__':
    # latitude, longitude, name, altitude, timezone

    coord = {
        'latitude': 24.4274827,
        'longitude': 54.6234876,
        'name': 'Masdar',
        'altitude': 0,
        'timezone': 'Asia/Dubai'
    }

    pv = PVGeneration(coordinates=coord)

    pv.configure_pv_system(n_arrays=5)
    pv_gen = pv.get_estimate_generation(year=2022)
    pv_pred = pv.get_forecast_generation(start='2022-02-13 00:00:00+04:00')
