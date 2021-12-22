from dataclasses import dataclass
from typing import Callable

import numpy as np
from alglib import GA, PSO, SA, VNS, AdaptiveSA
from alglib.param import AdaptiveSA_Param, GA_Param, PSO_Param, SA_Param, VNS_Param
from config import getConfig
from utils import CityDataLoader
from utils.tsp import twoOpt


def test_GA():
    algorithm = GA
    param = GA_Param(
        popNum=30, newPopRate=0.7, mutateProb=0.3, epochNum=30, maximize=False
    )

    schedule, value, dataloader = unitTest(algorithm, param)

    assert len(schedule) == len(dataloader)
    assert len(schedule) == len(set(schedule))
    assert np.isfinite(value)


def test_SA():
    methods = ["random", "best", "mixed"]
    for method in methods:
        algorithm = SA
        param = SA_Param(
            initialTemperate=(1 / 0.85) ** 20,
            terminatedTemperate=1e0,
            coolRate=0.85,
            epochNum=100,
            method=method,
            maximize=False,
            doIntersectAnalysis=False,
            initStatusJudgement=2 / 3,
        )

        schedule, value, dataloader = unitTest(algorithm, param)

        assert len(schedule) == len(dataloader)
        assert np.isfinite(value)


def test_AdaptiveSA():
    methods = ["random", "best", "mixed"]
    for method in methods:
        algorithm = AdaptiveSA
        param = AdaptiveSA_Param(
            minTemperate=1,
            penalWeight=1,
            delta=1,
            epochNum=2000,
            method=method,
            maximize=False,
            doIntersectAnalysis=False,
            initStatusJudgement=2 / 3,
        )

        schedule, value, dataloader = unitTest(algorithm, param)

        assert len(schedule) == len(dataloader)
        assert np.isfinite(value)


def test_VNS():
    algorithm = VNS
    param = VNS_Param(
        epochNum=1000,
        methods=[twoOpt, twoOpt],
        maximize=False,
        doIntersectAnalysis=True,
        initStatusJudgement=2 / 3,
    )

    schedule, value, dataloader = unitTest(algorithm, param)

    assert len(schedule) == len(dataloader)
    assert np.isfinite(value)


def test_PSO():
    algorithm = PSO
    param = PSO_Param(
        alpha=0.5,
        beta=0.7,
        particleNum=75,
        epochNum=200,
        maximize=False,
        doIntersectAnalysis=True,
        initStatusJudgement=2 / 3,
    )

    schedule, value, dataloader = unitTest(algorithm, param)

    assert len(schedule) == len(dataloader)
    assert np.isfinite(value)


def unitTest(algorithm: Callable, parameters: dataclass):
    config = getConfig()
    dataloader = CityDataLoader(config.cityInfo)
    alg = algorithm(parameters)
    schedule, value = alg(dataloader)

    return schedule, value, dataloader
