from dataclasses import dataclass
from typing import Callable, List

import numpy as np


@dataclass
class VariableNeighborhoodSearchParamSetting:
    epochNum: np.int16
    methods: List[Callable]
    maximize: bool
    doIntersectAnalysis: bool
    initStatusJudgement: float
