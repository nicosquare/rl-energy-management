import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':

    # General configurations

    plt.subplots_adjust(bottom=0.1, left=1.5, right=1.6, top=1.9)

    # Load data files

    consumption_m1 = pd.read_csv('resources/load_winter_30_d_s.csv', header=None)
    consumption_m2 = pd.read_csv('resources/load_winter_60_d_s.csv', header=None)
    consumption_m3 = pd.read_csv('resources/load_winter_school_s.csv', header=None)

    generation_m1 = pd.read_csv('resources/gen_winter_30_d_pv_s.csv', header=None)
    generation_m2 = pd.read_csv('resources/gen_winter_60_d_wind_s.csv', header=None)
    generation_m3 = pd.read_csv('resources/gen_winter_school_wind_s.csv', header=None)

    battery_parameters = pd.read_excel('resources/pb_acid_battery_parameters.xlsx')

    reference_power_week = pd.read_excel('resources/power_reference_weekly.xlsx')

    # Plot the data to check its content

    # Define time limits

    starting_day = 1
    starting_time = starting_day * 24 * 60
    ending_day = 5
    ending_time = ending_day * 24 * 60

    # Demand figures of microgrids

    fig, axs = plt.subplots(3, 1)

    color = 'tab:red'

    axs[0].set_ylabel("Power (kW) M1", color=color)
    axs[0].plot(consumption_m1[starting_time:ending_time], color=color)
    axs[0].tick_params(axis='y', labelcolor=color)

    axs[1].set_ylabel("Power (kW) M3", color=color)
    axs[1].plot(consumption_m2[starting_time:ending_time], color=color)
    axs[1].tick_params(axis='y', labelcolor=color)

    axs[2].set_xlabel("time (m)")
    axs[2].set_ylabel("Power (kW) M3", color=color)
    axs[2].plot(consumption_m3[starting_time:ending_time], color=color)
    axs[2].tick_params(axis='y', labelcolor=color)

    plt.show()
