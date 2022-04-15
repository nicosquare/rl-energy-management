import numpy as np

from gym import Env
from gym.spaces import Box

from src.components.microgrid import Microgrid

inf = np.float64('inf')


class MGSetGenerator(Env):

    def __init__(self, logging: bool = True):
        """
        Gym environment to simulate a Microgrid scenario
        """

        self.logging = logging
        self.state, self.reward, self.done, self.info = None, None, None, None

        """
        Observation space is composed by:
        
            soc: [0,1]
            ghi: [0, inf]
            pressure: [0, inf]
            wind_speed: [0, inf]
            air_temperature: [-inf, inf]
            relative_humidity = [0, inf]
        
        """

        self.observation_space = Box(
            low=np.float32(np.array([0.0, 0.0, 0.0, 0.0, -inf, 0])),
            high=np.float32(np.array([1.0, inf, inf, inf, inf, inf])),
            shape=(6,),
            dtype=np.float32
        )

        """
        Action space is a single continuous value defined by a Normal distribution with two parameters (actions):
            mu: Mean of the normal distribution
            sigma: Standard deviation of the normal distribution
        """

        self.action_space = Box(
            low=0,
            high=1,
            shape=(2,),
            dtype=np.float32
        )

        self.mg = Microgrid()

    def _observe(self):
        return self.mg.observe_by_setting_generator()

    def step(self, action):
        state, cost = self.mg.operation_by_setting_generator(power_rate=action.item(), logging=self.logging)
        self.state = state
        self.reward = -cost
        self.done = False
        self.info = {}

        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.mg.reset_current_step()
        return self._observe(), 0, False, {}

    def render(self, mode="human"):
        print('TODO')

    def set_logging(self, enabled: bool):
        self.logging = enabled

    def restore(self, time_step: int):
        self.mg.set_current_step(time_step=time_step)
