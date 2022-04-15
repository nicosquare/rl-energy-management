import numpy as np

from gym import Env
from gym.spaces import Box, Discrete

from src.components.microgrid import Microgrid

inf = np.float64('inf')


class MGSourceSelection(Env):

    def __init__(self):
        """
        Gym environment to simulate a Microgrid
        """
        self.state, self.reward, self.done, self.info = None, None, None, None

        self.observation_space = Box(
            low=-inf,
            high=inf,
            shape=(2,),
            dtype=np.float32
        )

        self.action_space = Discrete(8)

        self.mg = Microgrid()

    def _observe(self):
        return self.mg.observe_by_source_selection()

    def step(self, action):
        state, cost = self.mg.operation_by_source_selection(action=action)
        self.state = state
        self.reward = -cost
        self.done = False
        self.info = {}

        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.mg.reset_current_step()
        self.state, self.reward, self.done, self.info = self._observe(), 0, False, {}

    def render(self, mode="human"):
        # TODO: Create some render logic
        return None
