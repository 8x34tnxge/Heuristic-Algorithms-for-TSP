from abc import abstractmethod
from typing import Any, List, Tuple

import numpy as np


class Base(object):
    def __init__(self) -> None:
        self.localSchedule = []
        self.localValue = np.inf

        self.bestSchedule = []
        self.bestValue = np.inf

    @abstractmethod
    def run(self, *args: Any, **kwds: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Tuple[List[int], float]:
        self.run(*args, **kwds)
        return self.bestSchedule, self.bestValue
