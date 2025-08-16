import numpy as np
from abc import ABC, abstractmethod


class IAgent(ABC):
    @abstractmethod
    def observe(self, obs: np.ndarray) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def act(self) -> int:
        raise NotImplementedError()
    
    @abstractmethod
    def set_reward(self, reward: float) -> None:
        raise NotImplementedError()
