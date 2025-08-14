import numpy as np
from abc import ABC, abstractmethod


class ISim(ABC):
    @abstractmethod
    def tick(self, action: int) -> bool:
        ...

    @abstractmethod
    def to_image(self) -> np.ndarray:
        ...

