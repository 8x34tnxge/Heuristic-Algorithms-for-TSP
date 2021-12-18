from collections import defaultdict

import numpy as np
import pandas as pd

from alglib import SA, AdaptiveSA, VNS, PSO
from alglib.param import AdaptiveSA_Param, SA_Param, VNS_Param, PSO_Param
from config import getConfig
from utils import CityDataLoader, visualize
from utils.tsp import twoOpt


def main():
    VNS_test()

def PSO_test():
    eachTimes = 10
    result = defaultdict(list)
    algorithm = PSO
    param = PSO_Param(
        alpha = 0.5,
        beta = 0.7,
        particleNum = 75,
        epochNum = 200,
        maximize = False,
        doIntersectAnalysis = True,
        initStatusJudgement = 2 / 3,
    )

    for _ in range(eachTimes):
        value = unitTest(
            algorithm, param, "output", f"{algorithm.__name__}"
        )
        result[f"{algorithm.__name__}"].append(value)

    result = pd.DataFrame(
        np.array(
            [
                result[f"{algorithm.__name__}"]
            ]
        ),
        columns=[x for x in range(1, eachTimes + 1)],
        index=[f"{algorithm.__name__}"],
    )
    result["mean"] = result.mean(axis=1)
    result["std"] = result.std(axis=1)
    result = result.round(3)
    result.to_csv(f"testResult.csv", sep="|")

def VNS_test():
    eachTimes = 10
    result = defaultdict(list)
    algorithm = VNS
    param = VNS_Param(
        epochNum=1000,
        methods=[twoOpt],
        maximize=False,
        doIntersectAnalysis = True,
        initStatusJudgement = 2 / 3,
    )

    for _ in range(eachTimes):
        value = unitTest(
            algorithm, param, "output", f"{algorithm.__name__}"
        )
        result[f"{algorithm.__name__}"].append(value)

    result = pd.DataFrame(
        np.array(
            [
                result[f"{algorithm.__name__}"]
            ]
        ),
        columns=[x for x in range(1, eachTimes + 1)],
        index=[f"{algorithm.__name__}"],
    )
    result["mean"] = result.mean(axis=1)
    result["std"] = result.std(axis=1)
    result = result.round(3)
    result.to_csv(f"testResult.csv", sep="|")

def SA_test():
    eachTimes = 10
    algorithms = [SA, AdaptiveSA]
    result = defaultdict(list)
    methods = ["random", "best", "mixed"]

    for method in methods:
        params = [
            SA_Param(
                initialTemperate=(1 / 0.85) ** 20,
                terminatedTemperate=1e0,
                coolRate=0.85,
                epochNum=100,
                method=method,
                maximize=False,
                doIntersectAnalysis=False,
                initStatusJudgement = 2 / 3,
            ),
            AdaptiveSA_Param(
                minTemperate=1,
                penalWeight=1,
                delta=1,
                epochNum=2000,
                method=method,
                maximize=False,
                doIntersectAnalysis=False,
                initStatusJudgement = 2 / 3,
            ),
        ]
        for algorithm, param in zip(algorithms, params):
            for epoch in range(eachTimes):
                value = unitTest(
                    algorithm, param, "output", f"{algorithm.__name__}_{method}_{epoch}"
                )
                result[f"{algorithm.__name__}_{method}"].append(value)

    result = pd.DataFrame(
        np.array(
            [
                result[f"{key.__name__}_{method}"]
                for key in algorithms
                for method in methods
            ]
        ),
        columns=[x for x in range(1, eachTimes + 1)],
        index=[f"{key.__name__}_{method}" for key in algorithms for method in methods],
    )
    result["mean"] = result.mean(axis=1)
    result["std"] = result.std(axis=1)
    result = result.round(3)
    result.to_csv(f"testResult.csv", sep="|")


def unitTest(algorithm, param, outputDir, outputName):
    print("Loading infomation...")
    config = getConfig()
    dataloader = CityDataLoader(config.cityInfo)
    print("Init algorithm...")
    alg = algorithm(param)
    print("Begin to run the algorithm...")
    schedule, value = alg(dataloader)
    print("Running is completed!\n")

    try:
        assert np.isfinite(value)
    except AssertionError:
        raise NotImplementedError("Algorithm may broken")
    print("Result:")
    print("Route: " + "->".join(map(str, schedule)))
    print(f"Distance: {value}")
    # TODO visualize the result and attach detector to algorithm
    visualize(
        schedule,
        dataloader,
        alg.bestValueWatcher,
        fileDir=outputDir,
        fileName=outputName,
        save=True,
    )
    print("Result is saved!")
    return alg.bestValue


if __name__ == "__main__":
    main()
