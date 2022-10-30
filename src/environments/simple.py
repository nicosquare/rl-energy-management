import numpy as np

from gym import Env
from gym.spaces import Box

inf = np.float64('inf')

class SimpleEnv(Env):

    def __init__(self, batch_size: int = 1, n_actions: int = 1):

        """
            Initialization
        """

        self.batch_size = batch_size
        self.dim_action = n_actions

        self.time = 0    
        self.holding_cost = 0.5
        self.back_order_cost = 1.0
        self.dim_obs = 2 # tuple, (time, inventory)

        self.state = np.zeros([batch_size, n_actions, self.dim_obs])
        
        """
            Define a known demand (to know what should be the expected value)
        """

        self.means = np.array([[10 + i*10, 20 + i*10] for i in range(n_actions)]) # mean of the demand
        self.sigmas = np.array([[0.1, 0.2] for _ in range(n_actions)])# standard deviation for demand

        """
            Time: [0,2]
            Inventory: [0,100]
        """

        self.observation_space = Box(
            low=np.float32(np.array([0.0, 0.0])),
            high=np.float32(np.array([2.0, 100])),
            shape=(self.dim_obs,),
            dtype=np.float32
        )

        """
            Production: [-inf,inf]
        """

        self.action_space = Box(
            low=-0,
            high=100,
            shape=(n_actions,),
            dtype=np.float32
        )

    def reset(self):
        self.state = np.zeros([self.batch_size, self.dim_action, self.dim_obs])
        self.time = 0        
        return self.state, 0, {}, False

    def render(self, mode="human"):
        print('Rendering not defined yet')
    
    def step(self, action):

        curr_mean = np.ones((self.batch_size, 1)) * self.means[:,self.time]
        curr_sigma = np.ones((self.batch_size, 1)) * self.sigmas[:,self.time]
        
        demand = np.random.normal(loc=curr_mean, scale=curr_sigma)
        
        self.time += 1
        self.state[:,:,0] += 1 # time
        self.state[:,:,1] += action.reshape(demand.shape)-demand # inventory
        
        reward = -(  
            self.holding_cost * np.maximum(self.state[:,:,1], 0)
            +
            self.back_order_cost * np.maximum(-self.state[:,:,1], 0)
        )

        self.state[:,:,1] = np.maximum(self.state[:,:,1], 0) # I cannot have negative inventory

        return self.state + 0.0 , reward, {},  self.time >= 2