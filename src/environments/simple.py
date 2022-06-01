import numpy as np

from gym import Env
from gym.spaces import Box

inf = np.float64('inf')

class SimpleEnv(Env):

    def __init__(self, batch_size):

        """
            Initialization
        """

        self.batch_size = batch_size
        self.state = np.zeros([batch_size, 2])  # tuple, (time, inventory)

        self.time = 0    
        self.holding_cost = 0.5
        self.back_order_cost = 1.0

        """
            Define a known demand (to know what should be the expected value)
        """

        self.means=[10, 20] # mean of the demand
        self.sigmas=[1,2]    # standard deviation for demand

        """
            Time: [0,2]
            Inventory: [0,inf]
        """

        self.observation_space = Box(
            low=np.float32(np.array([0.0, 0.0])),
            high=np.float32(np.array([2.0, inf])),
            shape=(2,),
            dtype=np.float32
        )

        """
            Production: [-inf,inf]
        """

        self.action_space = Box(
            low=-inf,
            high=inf,
            shape=(1,),
            dtype=np.float32
        )

    def reset(self):
        self.state = np.zeros([self.batch_size, 2])
        self.time = 0        
        return self.state, 0, {}, False

    def render(self, mode="human"):
        print('Rendering not defined yet')
    
    def step(self, action):

        curr_mean = np.ones(self.batch_size)*self.means[self.time]
        curr_sigma = np.ones(self.batch_size)*self.sigmas[self.time]
        
        demand = np.random.normal(loc=curr_mean, scale=curr_sigma)
        
        self.time += 1
        self.state[:,0] += 1 # time
        self.state[:,1] += (action.cpu().numpy() - demand) # inventory
        
        reward = -(  
            self.holding_cost * np.maximum(self.state[:,1], 0)
            +
            self.back_order_cost * np.maximum(-self.state[:,1], 0)
        )

        self.state[:,1] = np.maximum(self.state[:,1], 0) # I cannot have negative inventory

        return self.state + 0.0 , reward, {},  self.time >= 2