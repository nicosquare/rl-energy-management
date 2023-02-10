import cvxpy
import numpy as np
from src.components.synthetic_house import SyntheticHouse
from src.environments.simple_microgrid import SimpleMicrogrid

from src.utils.tools import set_all_seeds, load_config

set_all_seeds(0)
def solver(house: SyntheticHouse, n: int = 24):

    battery = cvxpy.Variable(n+1)
    action = cvxpy.Variable(n)
    consumption = cvxpy.Variable(n)

    constraints = []

    # Battery
        # Starts in 0.1
    constraints.append(battery[0] == house.battery.soc_min)
        # Max and min batteries
    for i in range(n+1):
        constraints.append(battery[i] <= house.battery.soc_max)
        constraints.append(battery[i] >= house.battery.soc_min)


    # Action / Batteryn't

    for i in range(n):
        constraints.append(action[i] <= 1)
        constraints.append(action[i] >= -1)


    # Transition
    obj = 0

    for i in range(n):
        
        constraints.append(action[i] <= house.battery.p_charge_max)
        constraints.append(action[i] <= house.battery.p_discharge_max)
        # Update battery
        constraints.append(battery[i+1] == battery[i] + action[i] * house.battery.efficiency)
        # Update net 
        constraints.append(consumption[i] == house.demand[i]-house.pv_gen[i] + action[i] * house.battery.efficiency)


        obj += cvxpy.maximum(consumption[i] * (house.price[i] + house.emission[i]),0) 
        obj += cvxpy.maximum(-consumption[i] * house.price[i] * house.grid_sell_rate,0)  


    objective = cvxpy.Minimize(obj)
    prob = cvxpy.Problem(objective, constraints)
    res = prob.solve()

    return res, battery.value, action.value

def get_all_actions(env: SimpleMicrogrid, mode : str = 'train') -> np.ndarray:
    # Set mode, train, eval, test
    env.mg.change_mode(mode)
    state, reward, _, _ = env.reset()

    # Create arrays to hold score, battery SOC and Action for all houses
    rewards, battery_values, action_values = [],[],[]
    
    # Same for all houses
    for house in env.mg.houses:
        reward, batt, action = solver(house)
        
        rewards.append(reward)
        battery_values.append(batt)
        action_values.append(action)
    rewards = np.array(rewards)
    battery_values = np.array(battery_values)
    action_values = np.array(action_values)

    return rewards, battery_values, action_values

def loop_env(env: SimpleMicrogrid, action_values: np.ndarray, mode : str = 'train') -> np.ndarray:
    env.mg.change_mode(mode)

    state, reward, done, _ = env.reset()
    rewards = []
    time_step = 0

    # Cycle the entire episode with already computed actions by solver
    while not done:
        time_step = env.mg.current_step
        state,reward,done,_ = env.step(action_values[:,time_step])
        time_step += 1
        # Save rewards, mean over all batches
        rewards.append(reward.mean(axis=1))
    # Get the sum over steps and mean from 6 houses
    rewards = np.array(rewards).sum(axis=0).mean()
    # Calculate Score with actions
    return rewards, env.mg.get_houses_metrics() # output price, emissions ]
    

# set_all_seeds(0)
# # create environment,m save array of houses
# config = load_config("d_a2c_mgE1")
# config = config['train']
# env = SimpleMicrogrid(config=config['env'])

# # Train
# mode = 'train'
# rewards_t, battery_values, action_values = get_all_actions(env, mode)
# rewards_t_env, train_metrics = loop_env(env, action_values, mode)
# print("Mean rewards",rewards_t.mean(), rewards_t_env)
# print('train ', train_metrics)
