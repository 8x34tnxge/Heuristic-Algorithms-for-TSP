import random
from typing import Callable, List, Tuple

import numpy as np

from utils.base import DataLoader


def initSolution(dataLoader: DataLoader, distFunc: Callable):
    schedule = []
    for cityId, cityGeoInfo in dataLoader:
        schedule.append(cityId)
    random.shuffle(schedule)

    value = distFunc(schedule, dataLoader)

    return schedule, value


def twoOpt(
    self: object,
    schedule: List[int],
    value: float,
    dataLoader: DataLoader,
    maximized: bool = False,
) -> Tuple[List[int], float]:
    """the 2-opt method to generate new schedule

    Args:
        self (object): the algorithm object to fetch necessary params
        schedule (List[int]): the current schedule
        value (float): the current value
        dataLoader (DataLoader): where you can query the data from
        maximized (bool, optional): whether to maximize or minimize the target value. Defaults to False.

    Returns:
        Tuple[List[int], float]: new schedule and its value respectively
    """

    def calcDelta(schedule: List[int], prevPos: int, nextPos: int) -> float:
        """calculate the delta value between schedule before and after

        Args:
            schedule (List[int]): the current schedule
            prevPos (int): the 2-opt fore-position
            nextPos (int): the 2-opt post-position

        Returns:
            float: the delta value
        """
        if prevPos == nextPos:
            return 0
        if prevPos + nextPos + 1 == len(schedule):
            return 0

        prevGeoInfo4i = dataLoader[
            schedule[(prevPos + len(schedule) - 1) % len(schedule)]
        ]
        currGeoInfo4i = dataLoader[schedule[prevPos]]

        currGeoInfo4j = dataLoader[schedule[nextPos]]
        nextGeoInfo4j = dataLoader[schedule[(nextPos + 1) % len(schedule)]]

        delta = (
            np.linalg.norm(
                np.array(
                    [
                        prevGeoInfo4i.x - currGeoInfo4j.x,
                        prevGeoInfo4i.y - currGeoInfo4j.y,
                    ]
                )
            )
            + np.linalg.norm(
                np.array(
                    [
                        nextGeoInfo4j.x - currGeoInfo4i.x,
                        nextGeoInfo4j.y - currGeoInfo4i.y,
                    ]
                )
            )
            - np.linalg.norm(
                np.array(
                    [
                        prevGeoInfo4i.x - currGeoInfo4i.x,
                        prevGeoInfo4i.y - currGeoInfo4i.y,
                    ]
                )
            )
            - np.linalg.norm(
                np.array(
                    [
                        nextGeoInfo4j.x - currGeoInfo4j.x,
                        nextGeoInfo4j.y - currGeoInfo4j.y,
                    ]
                )
            )
        )
        return delta

    def getIdxViaRandom() -> Tuple[int, int]:
        """randomly fetch 2 index based on random strategy

        Returns:
            Tuple[int, int]: the fore-position and the post-position respectively
        """
        randomIdx1 = random.randint(0, len(schedule) - 1)
        randomIdx2 = random.randint(0, len(schedule) - 1)
        while randomIdx1 == randomIdx2 or randomIdx1 + randomIdx2 + 1 == len(schedule):
            randomIdx2 = random.randint(0, len(schedule) - 1)
        randomIdx1, randomIdx2 = sorted([randomIdx1, randomIdx2])
        return randomIdx1, randomIdx2

    def getIdxViaBest():
        """fetch 2 index based on best strategy

        Returns:
            Tuple[int, int]: the fore-position and the post-position respectively
        """
        delta = -np.inf
        bestIdx1, bestIdx2 = -1, -1
        for i in range(len(schedule) - 1):
            for j in range(i + 1, len(schedule)):
                if j - i + 1 == len(schedule) - 1:
                    continue

                if i + j + 1 == len(schedule):
                    continue

                currDelta = calcDelta(schedule, i, j)

                currDelta *= 1 if maximized else -1
                if currDelta > delta:
                    delta = currDelta
                    bestIdx1, bestIdx2 = i, j

        return bestIdx1, bestIdx2

    method = getattr(self.params, "method", "random")
    newSchedule = []
    newSchedule.extend(schedule)
    if method == "random":
        swapIdx1, swapIdx2 = getIdxViaRandom()
    elif method == "best":
        swapIdx1, swapIdx2 = getIdxViaBest()
    elif method == "mixed":
        swapIdx1, swapIdx2 = getIdxViaBest()
        delta = calcDelta(schedule, swapIdx1, swapIdx2)
        delta *= 1 if maximized else -1
        if delta <= 0:
            swapIdx1, swapIdx2 = getIdxViaRandom()
    else:
        raise KeyError(
            f"Methods only include 'random', 'best' and 'mixed'. Method {method} is not existed"
        )

    delta = calcDelta(schedule, swapIdx1, swapIdx2)
    value += delta
    for delta in range((swapIdx2 - swapIdx1) // 2 + 1):
        newSchedule[swapIdx1 + delta], newSchedule[swapIdx2 - delta] = (
            newSchedule[swapIdx2 - delta],
            newSchedule[swapIdx1 + delta],
        )

    return newSchedule, value


def distFunc(schedule: List[int], dataLoader: DataLoader) -> float:
    """the method to calc value based on the current schedule

    Args:
        schedule (List[int]): the current schedule
        dataLoader (DataLoader): where you can query the data from

    Returns:
        float: the target value
    """
    dist = 0
    for i in range(len(schedule)):
        prevGeoInfo = dataLoader[schedule[i]]
        nextGeoInfo = dataLoader[schedule[(i + 1) % len(schedule)]]

        deltaX = prevGeoInfo.x - nextGeoInfo.x
        deltaY = prevGeoInfo.y - nextGeoInfo.y
        dist += np.linalg.norm(np.array([deltaX, deltaY]))
    return dist
