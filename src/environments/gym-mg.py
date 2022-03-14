import numpy as np

from gym import Env
from gym.spaces import Box, MultiBinary

from src.components.microgrid import Microgrid


class GymMG(Env):

    def __init__(self):
        """
        Gym environment to simulate a Microgrid
        """
        self.state, self.reward, self.done, self.info = None, None, None, None

        self.observation_space = Box(
            low=-float(np.float('inf')),
            high=float(np.float('inf')),
            shape=(2, 1),
            dtype=np.float32
        )

        self.action_space = MultiBinary(3)

        self.mg = Microgrid()

    def step(self, action):
        load_t_next, pv_t_next, cost = self.mg.operate_one_step(action=action)
        self.state = np.array([load_t_next, pv_t_next])
        self.reward = -cost
        self.done = self.mg.get_current_step() == 24 * 365 # End of a year
        self.info = {}

        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.mg.reset_current_step()
        self.state, self.reward, self.done, self.info = None, None, None, None

    def render(self, mode="human"):
        print('TODO')

