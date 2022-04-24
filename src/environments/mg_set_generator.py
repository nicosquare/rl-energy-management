import numpy as np

from gym import Env
from gym.spaces import Box
from torch import Tensor

from src.components.microgrid import Microgrid, MicrogridArchitecture, MicrogridParameters

inf = np.float64('inf')


class MGSetGenerator(Env):

    def __init__(self, mg_arch: MicrogridArchitecture, mg_params: MicrogridParameters, batch_size: int = 1):
        """
        Gym environment to simulate a Microgrid scenario
        """

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
            fossil generator power: Value between 0 and 1 that indicates the rate of the nominal power it should use.
        """

        self.action_space = Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        self.mg = Microgrid(batch_size=batch_size, arch=mg_arch, params=mg_params)

    def _observe(self):
        return self.mg.observe_by_setting_generator()

    def step(self, action: Tensor):
        state, cost = self.mg.operation_by_setting_generator(power_rate=action)

        state = state
        reward = -cost
        done = False
        info = {}

        return state, reward, done, info

    def reset(self):
        self.mg.reset_current_step()
        return self._observe(), 0, False, {}

    def render(self, mode="human"):
        print('Rendering not defined yet')
