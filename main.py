from collections import defaultdict

import numpy as np
import pandas as pd

from algCore import SA, AdaptiveSA
from algCore.param import AdaptiveSA_Param, SA_Param
from config import getConfig
from utils import CityDataLoader, visualize


def main():
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
            ),
            AdaptiveSA_Param(
                minTemperate=1, penalWeight=1, delta=1, epochNum=2000, method=method
            ),
        ]
        for algorithm, param in zip(algorithms, params):
            for epoch in range(eachTimes):
                value = unitTest(
                    algorithm, param, f"{algorithm.__name__}_{method}_{epoch}"
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


def unitTest(algorithm, param, saveFileName):
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
        schedule, dataloader, alg.bestValueWatcher, fileName=saveFileName, save=True
    )
    print("Result is saved!")
    return alg.bestValue


if __name__ == "__main__":
    main()
