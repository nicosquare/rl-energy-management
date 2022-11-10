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


def plot_results(env, states, rewards, actions, net_energy, title, save=False, filename='results.png'):

    # Get list indexes

    quart_index = int(len(states) / 4)
    mid_index = int(len(states)/2)
    three_quart_index = int(len(states) * 3/4)
    last_index = len(states) - 1

    # Parse list of arrays

    rewards = np.stack(rewards, axis=0)
    actions = np.stack(actions, axis=0)
    states = np.stack(states, axis=0)
    net_energy = np.stack(net_energy, axis=0)

    all_socs = states[:,:,:,-1]

    # Plot results of interest

    fig = plt.figure(figsize=(10, 15))
    axs = fig.subplots(5, 2)

    for ax_x in axs:
        for ax_y in ax_x:
            ax_y.minorticks_on()
            ax_y.grid(True, which='both', axis='both', alpha=0.5)

    axs[0][0].plot(rewards[0,:,:].mean(axis=1), label='0%')
    axs[0][0].plot(rewards[quart_index,:,:].mean(axis=1), label='25%')
    axs[0][0].plot(rewards[mid_index,:,:].mean(axis=1), label='50%')
    axs[0][0].plot(rewards[three_quart_index,:,:].mean(axis=1), label='75%')
    axs[0][0].plot(rewards[last_index,:,:].mean(axis=1), label='100%')
    axs[0][0].set_title('Mean reward through time')
    axs[0][0].legend()

    axs[0][1].plot(rewards.sum(axis=1).mean(axis=1), label='Reward')
    axs[0][1].set_title('Mean reward through epochs')
    axs[0][1].legend()

    axs[1][0].plot(actions[0, :, :].mean(axis=1), label='0%')
    axs[1][0].plot(actions[quart_index, :, :].mean(axis=1), label='25%')
    axs[1][0].plot(actions[mid_index, :, :].mean(axis=1), label='50%')
    axs[1][0].plot(actions[three_quart_index, :, :].mean(axis=1), label='75%')
    axs[1][0].plot(actions[last_index, :, :].mean(axis=1), label='100%')
    axs[1][0].set_title('Mean action through time')
    axs[1][0].legend()

    axs[1][1].plot(actions.mean(axis=1).mean(axis=-1), label='Action')
    axs[1][1].legend()
    axs[1][1].set_title('Mean action through epochs')

    axs[2][0].plot(all_socs[0, :, :].mean(axis=1), label='0%')
    axs[2][0].plot(all_socs[quart_index, :, :].mean(axis=1), label='25%')
    axs[2][0].plot(all_socs[mid_index, :, :].mean(axis=1), label='50%')
    axs[2][0].plot(all_socs[three_quart_index, :, :].mean(axis=1), label='75%')
    axs[2][0].plot(all_socs[last_index, :, :].mean(axis=1), label='100%')
    axs[2][0].legend()
    axs[2][0].set_title('Mean SOC through time')

    axs[2][1].plot(all_socs.mean(axis=1).mean(axis=-1), label='SOC')
    axs[2][1].set_title('Mean SOC through epochs')
    axs[2][1].legend()

    axs[3][0].plot(net_energy[0, :, :].mean(axis=0), label='0%')
    axs[3][0].plot(net_energy[quart_index, :, :].mean(axis=0), label='25%')
    axs[3][0].plot(net_energy[mid_index, :, :].mean(axis=0), label='50%')
    axs[3][0].plot(net_energy[three_quart_index, :, :].mean(axis=0), label='75%')
    axs[3][0].plot(net_energy[last_index, :, :].mean(axis=0), label='100%')
    axs[3][0].plot((env.mg.demand - env.mg.pv_gen), label='Remaining', linestyle='--', linewidth=2)
    axs[3][0].legend()
    axs[3][0].set_title('Mean net energy through time')

    axs[3][1].plot(net_energy.mean(axis=1).sum(axis=-1), label='Net energy')
    axs[3][1].set_title('Mean net energy through epochs')
    axs[3][1].legend()

    axs[4][0].plot(env.mg.pv_gen, label='PV')
    axs[4][0].plot(env.mg.demand, label='Demand')
    axs[4][0].set_title('PV and Demand')
    axs[4][0].legend()

    axs[4][1].plot(env.mg.price, label='Price')
    axs[4][1].plot(env.mg.emission, label='Emission factor')
    axs[4][1].set_title('Price and Emission factor')
    axs[4][1].legend()
    
    fig.suptitle(title)
    fig.tight_layout()

    if save:
        plt.savefig(filename)

    plt.grid()
    plt.show()
