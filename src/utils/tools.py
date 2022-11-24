import yaml
import numpy as np
import torch
from os import path
from matplotlib import pyplot as plt

CONFIG_PATH = "config/"

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Function to load yaml configuration file

def load_config(config_name):
    config_name += ".yaml" 
    with open(path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


def plot_results(env, results, save : bool = False, filename: str ="results.png"):
    
    # Get epochs checkpoint indexes

    n_epochs = results['training_steps']
    quart_index = int(n_epochs / 4)
    mid_index = int(n_epochs/2)
    three_quart_index = int(n_epochs * 3/4)
    last_index = n_epochs - 1

    # Parse training data

    rewards = np.stack(results['train']['rewards'], axis=0)
    actions = np.stack(results['train']['actions'], axis=0)
    states = np.stack(results['train']['states'], axis=0)
    net_energy = np.stack(results['train']['net_energy'], axis=0)

    # Custom configuration depending on the experiment setup (number of houses/actions)

    if len(rewards.shape) == 4:
        single_house = True
        n_houses = 1
        all_socs = states[:,:,:,-1]
    else:
        single_house = False
        n_houses = env.n_houses
        all_socs = states[:,:,:,:,-1]

    # Parse metrics

    if not single_house:

        t_price_metric = np.stack(results['train']['price_metric'], axis=0)
        t_emissions_metric = np.stack(results['train']['emission_metric'], axis=0)
        e_price_metric = np.stack(results['eval']['price_metric'], axis=0)
        e_emissions_metric = np.stack(results['eval']['emission_metric'], axis=0)

        fig = plt.figure(figsize=(10, 15), num='Difference', constrained_layout=True)
        axs = fig.subplots(1,2)

        for ax in axs:
            ax.minorticks_on()
            ax.grid(True, which='both', axis='both', alpha=0.5)
        
        axs[0].plot(e_price_metric, label='Evaluation')
        axs[0].plot(t_price_metric, label='Training')
        axs[0].set_title('Price')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('$')
        axs[0].legend()

        plt.grid()

        axs[1].plot(t_emissions_metric, label='Training')
        axs[1].plot(e_emissions_metric, label='Evaluation')
        axs[1].set_title('Emissions')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('kgCO2')
        axs[1].legend()

    for ix_house in range(n_houses):

        house_name = f'House {ix_house + 1}'

        # Plot results of interest

        fig = plt.figure(figsize=(10, 15), num=house_name, constrained_layout=True)
        axs = fig.subplots(5, 2)

        for ax_x in axs:
            for ax_y in ax_x:
                ax_y.minorticks_on()
                ax_y.grid(True, which='both', axis='both', alpha=0.5)

        # Compute plots for the current house

        mean_reward_time = rewards.mean(axis=-2)
        mean_reward_epochs = rewards.sum(axis=1).mean(axis=-2)

        mean_action_time = actions.mean(axis=-1)
        mean_action_epochs = actions.sum(axis=1).mean(axis=-1)

        mean_soc_time = all_socs.mean(axis=-1)
        mean_soc_epochs = all_socs.mean(axis=1).mean(axis=-1)

        mean_net_energy_time = net_energy.mean(axis=-2)
        mean_net_energy_epochs = net_energy.mean(axis=-2).sum(axis=-1)

        demand = env.mg.demand if single_house else env.mg.houses[ix_house].demand
        pv_gen = env.mg.pv_gen if single_house else env.mg.houses[ix_house].pv_gen
        price = env.mg.price if single_house else env.mg.houses[ix_house].price
        emission = env.mg.emission if single_house else env.mg.houses[ix_house].emission

        # Plot results

        axs[0][0].plot(mean_reward_time[0,:,ix_house], label='0%')
        axs[0][0].plot(mean_reward_time[quart_index,:,ix_house], label='25%')
        axs[0][0].plot(mean_reward_time[mid_index,:,ix_house], label='50%')
        axs[0][0].plot(mean_reward_time[three_quart_index,:,ix_house], label='75%')
        axs[0][0].plot(mean_reward_time[last_index,:,ix_house], label='100%')
        axs[0][0].set_title('Mean reward through time')
        axs[0][0].legend(fontsize='xx-small')
        
        axs[0][1].plot(mean_reward_epochs[:,ix_house], label='Reward')
        axs[0][1].set_title('Mean accumulated reward through epochs')
        axs[0][1].legend(fontsize='xx-small')
        
        axs[1][0].plot(mean_action_time[0, :, None if single_house else ix_house], label='0%')
        axs[1][0].plot(mean_action_time[quart_index, :, None if single_house else ix_house], label='25%')
        axs[1][0].plot(mean_action_time[mid_index, :, None if single_house else ix_house], label='50%')
        axs[1][0].plot(mean_action_time[three_quart_index, :, None if single_house else ix_house], label='75%')
        axs[1][0].plot(mean_action_time[last_index, :, None if single_house else ix_house], label='100%')
        axs[1][0].set_title('Mean action through time')
        axs[1][0].legend(fontsize='xx-small')

        axs[1][1].plot(mean_action_epochs[:, None if single_house else ix_house], label='Action')
        axs[1][1].legend(fontsize='xx-small')
        axs[1][1].set_title('Mean action through epochs')

        axs[2][0].plot(mean_soc_time[0, :, None if single_house else ix_house], label='0%')
        axs[2][0].plot(mean_soc_time[quart_index, :, None if single_house else ix_house], label='25%')
        axs[2][0].plot(mean_soc_time[mid_index, :, None if single_house else ix_house], label='50%')
        axs[2][0].plot(mean_soc_time[three_quart_index, :, None if single_house else ix_house], label='75%')
        axs[2][0].plot(mean_soc_time[last_index, :, None if single_house else ix_house], label='100%')
        axs[2][0].legend(fontsize='xx-small')
        axs[2][0].set_title('Mean SOC through time')

        axs[2][1].plot(mean_soc_epochs[:,None if single_house else ix_house], label='SOC')
        axs[2][1].set_title('Mean SOC through epochs')
        axs[2][1].legend(fontsize='xx-small')

        axs[3][0].plot(mean_net_energy_time[0, None if single_house else ix_house, :].T, label='0%')
        axs[3][0].plot(mean_net_energy_time[quart_index, None if single_house else ix_house, :].T, label='25%')
        axs[3][0].plot(mean_net_energy_time[mid_index, None if single_house else ix_house, :].T, label='50%')
        axs[3][0].plot(mean_net_energy_time[three_quart_index, None if single_house else ix_house, :].T, label='75%')
        axs[3][0].plot(mean_net_energy_time[last_index, None if single_house else ix_house, :].T, label='100%')
        axs[3][0].plot((demand - pv_gen), label='Remaining', linestyle='--', linewidth=2)
        axs[3][0].legend(fontsize='xx-small')
        axs[3][0].set_title('Mean net energy through time')

        axs[3][1].plot(mean_net_energy_epochs[:, None if single_house else ix_house], label='Net energy')
        axs[3][1].set_title('Mean net energy through epochs')
        axs[3][1].legend(fontsize='xx-small')

        axs[4][0].plot(pv_gen, label='PV')
        axs[4][0].plot(demand, label='Demand')
        axs[4][0].set_title('PV and Demand', loc='left')
        axs[4][0].legend(fontsize='xx-small')

        axs[4][1].plot(price, label='Price')
        axs[4][1].plot(emission, label='Emission factor')
        axs[4][1].set_title('Price and Emission factor')
        axs[4][1].legend(fontsize='xx-small')

        if save:
            fig.savefig(f'{filename}_{house_name}.png', dpi=300)

    plt.show()
