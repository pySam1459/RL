import numpy as np
from abc import ABC, abstractmethod

from agents import IAgent


class ISim(ABC):
    """Simulation Interface which all simulation environments must inherit"""

    @abstractmethod
    def tick(self, action: int) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def to_image(self) -> np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def get_reward(self) -> float:
        raise NotImplementedError()


    def run(self, agent: IAgent, max_iter: int = -1) -> bool:
        _iter = 0
        while (_iter < max_iter or max_iter == -1):
            _iter += 1
            agent.observe(self.to_image())
            action = agent.act()

            terminal = self.tick(action)
            if terminal:
                break

        reward = self.get_reward()
        agent.set_reward(reward)
