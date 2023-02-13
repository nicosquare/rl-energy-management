import yaml
import numpy as np
import torch
import matplotlib as mpl

from os import path, getcwd
from matplotlib import pyplot as plt
from matplotlib import patches as patches


mpl.rcParams['figure.figsize'] = [10, 15]

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

def plot_metrics(metrics, save: bool = False, filename: str = "metrics"):
    
    # Parse metrics

    t_price_metric = np.stack(metrics['train']['agent']['price_metric'], axis=0)
    e_price_metric = np.stack(metrics['eval']['agent']['price_metric'], axis=0)
    tst_price_metric = np.stack(metrics['test']['agent']['price_metric'], axis=0)
    t_emissions_metric = np.stack(metrics['train']['agent']['emission_metric'], axis=0)
    e_emissions_metric = np.stack(metrics['eval']['agent']['emission_metric'], axis=0)
    tst_emissions_metric = np.stack(metrics['test']['agent']['emission_metric'], axis=0)

    fig = plt.figure(figsize=(15, 10), num='Difference', constrained_layout=True)
    axs = fig.subplots(1,2)

    for ax in axs:
        ax.minorticks_on()
        ax.grid(True, which='both', axis='both', alpha=0.5)
    
    axs[0].plot(t_price_metric, label='Training')
    axs[0].plot(e_price_metric, label='Evaluation')
    # axs[0].plot(tst_price_metric, label='Testing')
    axs[0].set_title('Price')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('$')

    axs[1].plot(t_emissions_metric, label='Training')
    axs[1].plot(e_emissions_metric, label='Evaluation')
    # axs[1].plot(tst_emissions_metric, label='Testing')
    axs[1].set_title('Emissions')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('kgCO2')

    # Patches to show the epochs where the grid profile changes

    prof_1_1 = patches.Rectangle((0, t_price_metric.min()), 1500, t_price_metric.max() - t_price_metric.min(), color='lightblue')
    prof_2_1 = patches.Rectangle((1500, t_price_metric.min()), 1500, t_price_metric.max() - t_price_metric.min(), color='lightgreen')
    prof_3_1 = patches.Rectangle((3000, t_price_metric.min()), 1500, t_price_metric.max() - t_price_metric.min(), color='lightgoldenrodyellow')
    prof_4_1 = patches.Rectangle((4500, t_price_metric.min()), 1500, t_price_metric.max() - t_price_metric.min(), color='lightpink')
    prof_5_1 = patches.Rectangle((6000, t_price_metric.min()), 1500, t_price_metric.max() - t_price_metric.min(), color='lightgray')
    prof_6_1 = patches.Rectangle((7500, t_price_metric.min()), 1500, t_price_metric.max() - t_price_metric.min(), color='lightsteelblue')

    prof_1_2 = patches.Rectangle((9000, t_price_metric.min()), 1500, t_price_metric.max() - t_price_metric.min(), color='lightblue')
    prof_2_2 = patches.Rectangle((10500, t_price_metric.min()), 1500, t_price_metric.max() - t_price_metric.min(), color='lightgreen')
    prof_3_2 = patches.Rectangle((12000, t_price_metric.min()), 1500, t_price_metric.max() - t_price_metric.min(), color='lightgoldenrodyellow')
    prof_4_2 = patches.Rectangle((13500, t_price_metric.min()), 1500, t_price_metric.max() - t_price_metric.min(), color='lightpink')
    prof_5_2 = patches.Rectangle((15000, t_price_metric.min()), 1500, t_price_metric.max() - t_price_metric.min(), color='lightgray')
    prof_6_2 = patches.Rectangle((16500, t_price_metric.min()), 1500, t_price_metric.max() - t_price_metric.min(), color='lightsteelblue')

    # Add all the patches to the plot

    axs[0].add_patch(prof_1_1)
    axs[0].add_patch(prof_2_1)
    axs[0].add_patch(prof_3_1)
    axs[0].add_patch(prof_4_1)
    axs[0].add_patch(prof_5_1)
    axs[0].add_patch(prof_6_1)
    
    axs[0].add_patch(prof_1_2)
    axs[0].add_patch(prof_2_2)
    axs[0].add_patch(prof_3_2)
    axs[0].add_patch(prof_4_2)
    axs[0].add_patch(prof_5_2)
    axs[0].add_patch(prof_6_2)

    prof_1_1 = patches.Rectangle((0, t_emissions_metric.min()), 1500, t_emissions_metric.max() - t_emissions_metric.min(), color='lightblue')
    prof_2_1 = patches.Rectangle((1500, t_emissions_metric.min()), 1500, t_emissions_metric.max() - t_emissions_metric.min(), color='lightgreen')
    prof_3_1 = patches.Rectangle((3000, t_emissions_metric.min()), 1500, t_emissions_metric.max() - t_emissions_metric.min(), color='lightgoldenrodyellow')
    prof_4_1 = patches.Rectangle((4500, t_emissions_metric.min()), 1500, t_emissions_metric.max() - t_emissions_metric.min(), color='lightpink')
    prof_5_1 = patches.Rectangle((6000, t_emissions_metric.min()), 1500, t_emissions_metric.max() - t_emissions_metric.min(), color='lightgray')
    prof_6_1 = patches.Rectangle((7500, t_emissions_metric.min()), 1500, t_emissions_metric.max() - t_emissions_metric.min(), color='lightsteelblue')

    prof_1_2 = patches.Rectangle((9000, t_emissions_metric.min()), 1500, t_emissions_metric.max() - t_emissions_metric.min(), color='lightblue')
    prof_2_2 = patches.Rectangle((10500, t_emissions_metric.min()), 1500, t_emissions_metric.max() - t_emissions_metric.min(), color='lightgreen')
    prof_3_2 = patches.Rectangle((12000, t_emissions_metric.min()), 1500, t_emissions_metric.max() - t_emissions_metric.min(), color='lightgoldenrodyellow')
    prof_4_2 = patches.Rectangle((13500, t_emissions_metric.min()), 1500, t_emissions_metric.max() - t_emissions_metric.min(), color='lightpink')
    prof_5_2 = patches.Rectangle((15000, t_emissions_metric.min()), 1500, t_emissions_metric.max() - t_emissions_metric.min(), color='lightgray')
    prof_6_2 = patches.Rectangle((16500, t_emissions_metric.min()), 1500, t_emissions_metric.max() - t_emissions_metric.min(), color='lightsteelblue')

    axs[1].add_patch(prof_1_1)
    axs[1].add_patch(prof_2_1)
    axs[1].add_patch(prof_3_1)
    axs[1].add_patch(prof_4_1)
    axs[1].add_patch(prof_5_1)
    axs[1].add_patch(prof_6_1)

    axs[1].add_patch(prof_1_2)
    axs[1].add_patch(prof_2_2)
    axs[1].add_patch(prof_3_2)
    axs[1].add_patch(prof_4_2)
    axs[1].add_patch(prof_5_2)
    axs[1].add_patch(prof_6_2)


    # Review if cvxpy metrics are available
    
    if 'cvxpy' in metrics['train']:
        t_cvxpy_price_metric = np.stack(metrics['train']['cvxpy']['price_metric'], axis=0)
        t_cvxpy_emissions_metric = np.stack(metrics['train']['cvxpy']['emission_metric'], axis=0)

        axs[0].plot(t_cvxpy_price_metric, label='CVXPY Training')
        axs[1].plot(t_cvxpy_emissions_metric, label='CVXPY Training')

    if 'cvxpy' in metrics['eval']:
        e_cvxpy_price_metric = np.stack(metrics['eval']['cvxpy']['price_metric'], axis=0)
        e_cvxpy_emissions_metric = np.stack(metrics['eval']['cvxpy']['emission_metric'], axis=0)

        axs[0].plot(e_cvxpy_price_metric, label='CVXPY Evaluation')
        axs[1].plot(e_cvxpy_emissions_metric, label='CVXPY Evaluation')

    if 'cvxpy' in metrics['test']:
        tst_cvxpy_price_metric = np.stack(metrics['test']['cvxpy']['price_metric'], axis=0)
        tst_cvxpy_emissions_metric = np.stack(metrics['test']['cvxpy']['emission_metric'], axis=0)

        axs[0].plot(tst_cvxpy_price_metric, label='CVXPY Testing')
        axs[1].plot(tst_cvxpy_emissions_metric, label='CVXPY Testing')
    

    axs[0].legend()
    axs[1].legend()

    if save:
        fig.savefig(f'{filename}.png', dpi=300)

    plt.show()

def plot_rollout(env, results, save : bool = False, filename: str ="results"):
    
    # Get epochs checkpoint indexes

    n_epochs = results['training_steps']
    quart_index = int(n_epochs / 4)
    mid_index = int(n_epochs/2)
    three_quart_index = int(n_epochs * 3/4)
    last_index = n_epochs - 1

    # Parse training data

    rewards = np.stack(results['train']['agent']['rewards'], axis=0)
    actions = np.stack(results['train']['agent']['actions'], axis=0)
    states = np.stack(results['train']['agent']['states'], axis=0)
    net_energy = np.stack(results['train']['agent']['net_energy'], axis=0)

    # Custom configuration depending on the experiment setup (number of houses/actions)

    if len(rewards.shape) == 4:
        single_house = True
        n_houses = 1
        all_socs = states[:,:,:,-1]
    else:
        single_house = False
        n_houses = env.n_houses
        all_socs = states[:,:,:,:,-1]

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
