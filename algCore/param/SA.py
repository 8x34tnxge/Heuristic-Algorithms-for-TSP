import numpy as np
from dataclasses import dataclass

@dataclass
class SimulatedAnnealingParamSetting():
    initialTemperate: np.float32
    terminatedTemperate: np.float32
    coolRate: np.float32
    epochNum: np.int16
    method: str

@dataclass
class AdaptiveSimulatedAnnealingParamSetting():
    epochNum: np.int16
    minTemperate: np.float32
    penalWeight: np.float32
    delta: np.float32
    method: str