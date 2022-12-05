import cvxpy
import traceback
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from src.components.synthetic_house import SyntheticHouse
from src.components.synthetic_microgrid import SyntheticMicrogrid
from src.environments.simple_microgrid import SimpleMicrogrid
from src.rl.a2c.d_simple_microgrid import Agent
from src.components.battery import Battery, BatteryParameters
from src.utils.tools import set_all_seeds, load_config, plot_metrics

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

def get_all_actions(env: SimpleMicrogrid, mode : str = 'train', n : int = 24) -> np.ndarray:
    # Set mode, train, eval, test
    env.mg.change_mode(mode)

    # Create arrays to hold score, battery SOC and Action for all houses
    rewards, battery_values, action_values = [],[],[]
    
    # Same for all houses
    for house in env.mg.houses:
        reward, batt, action = solver(house, n)
        
        rewards.append(reward)
        battery_values.append(batt)
        action_values.append(action)
    battery_values = np.array(battery_values)
    action_values = np.array(action_values)
    rewards = np.array(rewards)

    return rewards, battery_values, action_values
    # print("Mean", scores.mean(), "\n Scores",rewards)    

def loop_env(env: SimpleMicrogrid, action_values: np.ndarray, mode : str = 'train') -> np.ndarray:
    done = 0
    env.mg.change_mode(mode)

    env.reset()

    # Cycle the entire episode with already computed actions by solver
    while not done:
        time_step = env.mg.current_step
        _,_,done,_ = env.step(action_values[:,time_step])
        time_step += 1

    # Calculate Score with actions
    return env.mg.get_houses_metrics() # output price, emissions 


"""
    Main method definition
"""

if __name__ == '__main__':
    try:
        # Create environment
        set_all_seeds(0)

        # AGENT
        model = "d_a2c_mg"
        config = load_config(model)
        config = config['train']
        n = config['env']['rollout_steps']
        my_env = SimpleMicrogrid(config=config['env'])

        agent = Agent(env=my_env, config = config)
        metrics = agent.train()
        metrics['test'] = agent.test()


        # CVXPY
        config = load_config("zero_mg")
        env = SimpleMicrogrid(config=config['env'])

        # Train
        mode = 'train'
            # Calculate best actionjs
        rewards, battery_values, action_values = get_all_actions(env, mode, n)
            # Apply actions to environment
        train_metrics = loop_env(env, action_values, mode)
            # Create empty dictionary 
        metrics['train']['cvxpy'] = {}
            # Create a list of same metric so it appears as a line in the graph 
        metrics['train']['cvxpy']['price_metric'] = [train_metrics[0].mean() for _ in range(agent.training_steps)]
        metrics['train']['cvxpy']['emission_metric'] = [train_metrics[1].mean() for _ in range(agent.training_steps)]
        # print('train ', rewards[0].me an())
        # print('train ', train_metrics)


        # Eval
        mode = 'eval'
        rewards, battery_values, action_values = get_all_actions(env, mode, n)
        eval_metrics = loop_env(env, action_values, mode)
            # Create empty dictionary 
        metrics['eval']['cvxpy'] = {}
        metrics['eval']['cvxpy']['price_metric'] =  [eval_metrics[0].mean() for _ in range(agent.training_steps)]
        metrics['eval']['cvxpy']['emission_metric'] = [eval_metrics[1].mean() for _ in range(agent.training_steps)]
        # print('Eval ', rewards[0].mean())
        # print('eval ', eval_metrics)

        # Test
        mode = 'test'
        rewards, battery_values, action_values = get_all_actions(env, mode, n)
        test_metrics = loop_env(env, action_values, mode)

        # Create empty dictionary 
        metrics['test']['cvxpy'] = {}
        metrics['test']['cvxpy']['price_metric'] = [test_metrics[0].mean() for _ in range(agent.training_steps)]
        metrics['test']['cvxpy']['emission_metric'] = [test_metrics[1].mean() for _ in range(agent.training_steps)]

        
        filename = "metrics_" + datetime.today().strftime("%H_%M_%S__%d_%m_%Y")
        plot_metrics(metrics=metrics, save=True, filename=filename)

        agent.wdb_logger.finish()

    except (RuntimeError, KeyboardInterrupt):

        traceback.print_exc()