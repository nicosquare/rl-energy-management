import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def generate_pv_normal(n_days=1, peak_power=1):
    # Build a normal for one day with 1 min resolution

    one_day_hours = 24 * 60
    one_day_mu = one_day_hours / 2
    one_day_sigma = one_day_hours / 8

    steps_one_day = np.array(range(0, one_day_hours))
    pv_power_one_day = 1 / (one_day_sigma * 60 * np.sqrt(2 * np.pi)) * np.exp(
        - (steps_one_day - one_day_mu) ** 2 / (2 * one_day_sigma ** 2))
    scale_factor = peak_power / np.max(pv_power_one_day)
    pv_power_one_day *= scale_factor

    # Build a vector repeating one day according to the required time steps

    pv_power_complete = np.array(pv_power_one_day)

    for day in range(n_days - 1):
        pv_power_complete = np.concatenate((pv_power_complete, pv_power_one_day))

    return pv_power_complete


def generate_gas_plant(n_days=1, peak_power=1):
    # Define the time limits

    days_hours = n_days * 24 * 60

    # Gas generation tend to be constant for a day

    return peak_power * np.ones(days_hours)


def get_load_from_dataset(n_days=1):
    # Define the time limit

    days_hours = n_days * 24 * 60

    # Assume the energy consumption like the one in the UK dataset for winter

    return pd.read_csv('../resources/load_winter_30_d_s.csv', header=None)[0:days_hours - 1]


if __name__ == '__main__':
    # Define data parameters

    sim_days = 2
    sim_pv_power = 30000  # kW
    sim_gas_power = 20000  # kW

    pv_power = generate_pv_normal(n_days=sim_days, peak_power=sim_pv_power)
    gas_power = generate_gas_plant(n_days=sim_days, peak_power=sim_gas_power)
    load = get_load_from_dataset(n_days=sim_days)

    plt.plot(range(len(pv_power)), pv_power)
    plt.plot(range(len(gas_power)), gas_power)
    plt.plot(range(len(load)), load)
    plt.show()
