import numpy as np

from gym import Env
from gym.spaces import Box

from src.components.microgrid import Microgrid


class MGSetGenerator(Env):

    def __init__(self):
        """
        Gym environment to simulate a Microgrid
        """
        self.state, self.reward, self.done, self.info = None, None, None, None

        self.observation_space = Box(
            low=-float(-np.float('inf')),
            high=float(np.float('inf')),
            shape=(6,),
            dtype=np.float32
        )

        self.action_space = Box(
            low=-float(-np.float('inf')),
            high=float(np.float('inf')),
            shape=(1,),
            dtype=np.float32
        )

        self.mg = Microgrid()

    def _observe(self):
        return self.mg.observe_by_setting_generator()

    def step(self, action):
        state, cost = self.mg.operation_by_setting_generator(power_rate=action)
        self.state = state
        self.reward = -cost
        self.done = self.mg.get_current_step() == 24 * 365 - 1  # End of a year
        self.info = {}

        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.mg.reset_current_step()
        self.state, self.reward, self.done, self.info = None, None, None, None

    def render(self, mode="human"):
        print('TODO')
