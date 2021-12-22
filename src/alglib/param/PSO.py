from dataclasses import dataclass

import numpy as np


@dataclass
class ParticleSwarmOptimizationAlgorithmParamSetting:
    alpha: float
    beta: float
    particleNum: np.int16
    epochNum: np.int16
    maximize: bool
    doIntersectAnalysis: bool
    initStatusJudgement: float
