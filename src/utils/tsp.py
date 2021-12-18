import random
from typing import Callable, List, Tuple, Dict

import numpy as np

from utils.base import DataLoader


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


def intersectionAnalysis(
    schedule: List[int], dataLoader: DataLoader
) -> Tuple[int, int] or None:
    """check whether the schedule is intersected

    Args:
        schedule (List[int]): the current schedule
        dataLoader (DataLoader): where you can query the data from

    Returns:
        Tuple[int, int] or None: the two key points in sequence or nothing
    """

    def isIntersected(points: List[Dict[str, float]]):
        """check four points for intersection

        Reference:
        https://blog.csdn.net/rickliuxiao/article/details/6259322?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.compare&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.compare

        Args:
            points (List[Dict[str, float]]): four points in sequence

        Returns:
            bool: whether is intersected
        """
        func = lambda v1, v2, v3, v4: v1 * v4 - v2 * v3
        delta = func(
            points[1].x - points[0].x,
            points[2].x - points[3].x,
            points[1].y - points[0].y,
            points[2].y - points[3].y,
        )
        if -(1e-6) <= delta <= 1e-6:
            return False
        alpha = func(
            points[2].x - points[0].x,
            points[2].x - points[3].x,
            points[2].y - points[0].y,
            points[2].y - points[3].y,
        )
        if alpha / delta > 1 or alpha / delta < 0:
            return False
        beta = func(
            points[1].x - points[0].x,
            points[2].x - points[0].x,
            points[1].y - points[0].y,
            points[2].y - points[0].y,
        )
        if beta / delta > 1 or beta / delta < 0:
            return False
        return True

    for prevIdx, prevPos in enumerate(schedule[:-1]):
        for postIdx, postPos in enumerate(schedule):
            if postIdx - prevIdx <= 1 or (
                prevIdx + postIdx == len(schedule) - 1 and prevIdx * postIdx == 0
            ):
                continue

            if isIntersected(
                [
                    dataLoader[prevPos],
                    dataLoader[schedule[prevIdx + 1]],
                    dataLoader[postPos],
                    dataLoader[schedule[(postIdx + 1) % len(dataLoader)]],
                ]
            ):
                return schedule[prevIdx + 1], postPos

    return None


def intersectionRefactor(
    schedule: List[int], dataLoader: DataLoader, value, prevPos: int, postPos: int
) -> Tuple[List[int], float]:
    """the method to refactor schedule which contains intersection

    Args:
        schedule (List[int]): the current schedule
        dataLoader (DataLoader): where you can query the data from
        prevPos (int): previous key point position
        postPos (int): post key point position

    Returns:
        Tuple[List[int], float]: new schedule and its value respectively
    """

    def calcDelta(
        schedule: List[int], dataLoader: DataLoader, prevPos: int, postPos: int
    ) -> float:
        prevIdx, postIdx = schedule.index(prevPos), schedule.index(postPos)
        delta = (
            -np.linalg.norm(
                np.array(
                    [
                        dataLoader[prevPos].x - dataLoader[schedule[prevIdx - 1]].x,
                        dataLoader[prevPos].y - dataLoader[schedule[prevIdx - 1]].y,
                    ]
                )
            )
            - np.linalg.norm(
                np.array(
                    [
                        dataLoader[postPos].x
                        - dataLoader[schedule[(postIdx + 1) % len(schedule)]].x,
                        dataLoader[postPos].y
                        - dataLoader[schedule[(postIdx + 1) % len(schedule)]].y,
                    ]
                )
            )
            + np.linalg.norm(
                np.array(
                    [
                        dataLoader[prevPos].x
                        - dataLoader[schedule[(postIdx + 1) % len(schedule)]].x,
                        dataLoader[prevPos].y
                        - dataLoader[schedule[(postIdx + 1) % len(schedule)]].y,
                    ]
                )
            )
            + np.linalg.norm(
                np.array(
                    [
                        dataLoader[postPos].x - dataLoader[schedule[prevIdx - 1]].x,
                        dataLoader[postPos].y - dataLoader[schedule[prevIdx - 1]].y,
                    ]
                )
            )
        )
        return delta

    prevIdx, postIdx = schedule.index(prevPos), schedule.index(postPos)
    value += calcDelta(schedule, dataLoader, prevPos, postPos)
    schedule[prevIdx : postIdx + 1] = reversed(schedule[prevIdx : postIdx + 1])
    return schedule, value


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
    initStatus: bool = False,
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

    value += calcDelta(schedule, swapIdx1, swapIdx2)
    for delta in range((swapIdx2 - swapIdx1) // 2 + 1):
        newSchedule[swapIdx1 + delta], newSchedule[swapIdx2 - delta] = (
            newSchedule[swapIdx2 - delta],
            newSchedule[swapIdx1 + delta],
        )

    while not initStatus and self.params.doIntersectAnalysis:
        analysisResult = intersectionAnalysis(newSchedule, dataLoader)
        if analysisResult is None:
            break
        prevPos, postPos = analysisResult
        newSchedule, value = intersectionRefactor(
            newSchedule, dataLoader, value, prevPos, postPos
        )

    return newSchedule, value


def swap(
    self: object,
    schedule: List[int],
    value: float,
    dataLoader: DataLoader,
    maximized: bool = False,
    initStatus: bool = False,
) -> Tuple[List[int], float]:
    def calcSwapDelta(schedule, dataLoader, prevIdx, postIdx):
        prevIdx, postIdx = sorted([prevIdx, postIdx])
        delta = (
            -np.linalg.norm(
                np.array(
                    [
                        dataLoader[schedule[prevIdx]].x
                        - dataLoader[schedule[prevIdx - 1]].x,
                        dataLoader[schedule[prevIdx]].y
                        - dataLoader[schedule[prevIdx - 1]].y,
                    ]
                )
            )
            - np.linalg.norm(
                np.array(
                    [
                        dataLoader[schedule[postIdx]].x
                        - dataLoader[schedule[(postIdx + 1) % len(schedule)]].x,
                        dataLoader[schedule[postIdx]].y
                        - dataLoader[schedule[(postIdx + 1) % len(schedule)]].y,
                    ]
                )
            )
            + np.linalg.norm(
                np.array(
                    [
                        dataLoader[schedule[prevIdx]].x
                        - dataLoader[schedule[(postIdx + 1) % len(schedule)]].x,
                        dataLoader[schedule[prevIdx]].y
                        - dataLoader[schedule[(postIdx + 1) % len(schedule)]].y,
                    ]
                )
            )
            + np.linalg.norm(
                np.array(
                    [
                        dataLoader[schedule[postIdx]].x
                        - dataLoader[schedule[prevIdx - 1]].x,
                        dataLoader[schedule[postIdx]].y
                        - dataLoader[schedule[prevIdx - 1]].y,
                    ]
                )
            )
        )

        if prevIdx == 0 and postIdx == len(dataLoader) - 1:
            delta = (
                -np.linalg.norm(
                    np.array(
                        [
                            dataLoader[schedule[prevIdx]].x
                            - dataLoader[schedule[(prevIdx + 1) % len(schedule)]].x,
                            dataLoader[schedule[prevIdx]].y
                            - dataLoader[schedule[(prevIdx + 1) % len(schedule)]].y,
                        ]
                    )
                )
                - np.linalg.norm(
                    np.array(
                        [
                            dataLoader[schedule[postIdx]].x
                            - dataLoader[schedule[postIdx - 1]].x,
                            dataLoader[schedule[postIdx]].y
                            - dataLoader[schedule[postIdx - 1]].y,
                        ]
                    )
                )
                + np.linalg.norm(
                    np.array(
                        [
                            dataLoader[schedule[prevIdx]].x
                            - dataLoader[schedule[postIdx - 1]].x,
                            dataLoader[schedule[prevIdx]].y
                            - dataLoader[schedule[postIdx - 1]].y,
                        ]
                    )
                )
                + np.linalg.norm(
                    np.array(
                        [
                            dataLoader[schedule[postIdx]].x
                            - dataLoader[schedule[(prevIdx + 1) % len(schedule)]].x,
                            dataLoader[schedule[postIdx]].y
                            - dataLoader[schedule[(prevIdx + 1) % len(schedule)]].y,
                        ]
                    )
                )
            )
        elif not (abs(prevIdx - postIdx) <= 1):
            delta += (
                -np.linalg.norm(
                    np.array(
                        [
                            dataLoader[schedule[prevIdx]].x
                            - dataLoader[schedule[(prevIdx + 1) % len(schedule)]].x,
                            dataLoader[schedule[prevIdx]].y
                            - dataLoader[schedule[(prevIdx + 1) % len(schedule)]].y,
                        ]
                    )
                )
                - np.linalg.norm(
                    np.array(
                        [
                            dataLoader[schedule[postIdx]].x
                            - dataLoader[schedule[postIdx - 1]].x,
                            dataLoader[schedule[postIdx]].y
                            - dataLoader[schedule[postIdx - 1]].y,
                        ]
                    )
                )
                + np.linalg.norm(
                    np.array(
                        [
                            dataLoader[schedule[prevIdx]].x
                            - dataLoader[schedule[postIdx - 1]].x,
                            dataLoader[schedule[prevIdx]].y
                            - dataLoader[schedule[postIdx - 1]].y,
                        ]
                    )
                )
                + np.linalg.norm(
                    np.array(
                        [
                            dataLoader[schedule[postIdx]].x
                            - dataLoader[schedule[(prevIdx + 1) % len(schedule)]].x,
                            dataLoader[schedule[postIdx]].y
                            - dataLoader[schedule[(prevIdx + 1) % len(schedule)]].y,
                        ]
                    )
                )
            )

        return delta

    alpha = getattr(self.params, "alpha", 0.5)
    beta = getattr(self.params, "beta", 0.5)
    operations = []
    for idx, pos in enumerate(schedule):
        if schedule[idx] != self.localSchedule[idx]:
            operations.append((idx, self.localSchedule.index(pos), alpha))
        if schedule[idx] != self.bestSchedule[idx]:
            operations.append((idx, self.bestSchedule.index(pos), beta))

    for operation in operations:
        prevIdx, postIdx, probability = operation
        if random.random() < probability:
            value += calcSwapDelta(schedule, dataLoader, prevIdx, postIdx)
            schedule[prevIdx], schedule[postIdx] = schedule[postIdx], schedule[prevIdx]

    while not initStatus and self.params.doIntersectAnalysis:
        analysisResult = intersectionAnalysis(schedule, dataLoader)
        if analysisResult is None:
            break
        prevPos, postPos = analysisResult
        schedule, value = intersectionRefactor(
            schedule, dataLoader, value, prevPos, postPos
        )

    return schedule, value
