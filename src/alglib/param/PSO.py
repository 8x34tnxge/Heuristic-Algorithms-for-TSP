from dataclasses import dataclass
from typing import Callable, List

import numpy as np


@dataclass
class ParticleSwarmOptimizationAlgorithmParamSetting:
    alpha: float
    beta: float
    particleNum: np.int16
    epochNum: np.int16
    maximize: bool
