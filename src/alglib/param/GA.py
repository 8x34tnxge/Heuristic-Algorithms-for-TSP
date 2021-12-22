from dataclasses import dataclass

import numpy as np


@dataclass
class GeneticAlgorithmParamSetting:
    popNum: np.int16
    newPopRate: float
    mutateProb: float
    epochNum: np.int16
    maximize: bool
