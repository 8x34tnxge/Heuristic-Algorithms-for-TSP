import random
from typing import Callable, Dict, List, Tuple

import numpy as np

from utils.dataloader import CityDataLoader


def distFunc(schedule: List[int], dataLoader: CityDataLoader) -> float:
    """the method to calc value based on the current schedule

    Args:
        schedule (List[int]): the current schedule
        dataLoader (DataLoader): where you can query the data from

    Returns:
        float: the target value
    """
    dist = 0
    for idx, pos in enumerate(schedule):
        dist += dataLoader.distMat[pos, schedule[(idx + 1) % len(schedule)]]

    return dist


def intersectionAnalysis(
    schedule: List[int], dataLoader: CityDataLoader
) -> Tuple[int, int] or None:
    """check whether the schedule is intersected

    Args:
        schedule (List[int]): the current schedule
        dataLoader (CityDataLoader): where you can query the data from

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
                return prevIdx + 1, postIdx

    return None


def intersectionRefactor(
    schedule: List[int], dataLoader: CityDataLoader, value, prevIdx: int, postIdx: int
) -> Tuple[List[int], float]:
    """the method to refactor schedule which contains intersection

    Args:
        schedule (List[int]): the current schedule
        dataLoader (CityDataLoader): where you can query the data from
        prevIdx (int): the index of previous key point
        postIdx (int): the index of post key point

    Returns:
        Tuple[List[int], float]: new schedule and its value respectively
    """

    def calcDelta(
        schedule: List[int], dataLoader: CityDataLoader, prevPos: int, postPos: int
    ) -> float:
        prevIdx, postIdx = schedule.index(prevPos), schedule.index(postPos)
        delta = (
            -dataLoader.distMat[prevPos, schedule[prevIdx - 1]]
            - dataLoader.distMat[postPos, schedule[(postIdx + 1) % len(schedule)]]
            + dataLoader.distMat[prevPos, schedule[(postIdx + 1) % len(schedule)]]
            + dataLoader.distMat[postPos, schedule[prevIdx - 1]]
        )
        return delta

    value += calcDelta(schedule, dataLoader, schedule[prevIdx], schedule[postIdx])
    schedule[prevIdx : postIdx + 1] = reversed(schedule[prevIdx : postIdx + 1])
    return schedule, value


def initSolution(
    dataLoader: CityDataLoader, distFunc: Callable
) -> Tuple[Tuple[List[int], float]]:
    """the method to initialize the solution

    Args:
        dataLoader (CityDataLoader): where you can query the data from
        distFunc (Callable): the method to calc total value for solution

    Returns:
        Tuple[List[int], float]: the schedule and its value respectively

    """
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
    dataLoader: CityDataLoader,
    maximized: bool = False,
    initStatus: bool = False,
) -> Tuple[List[int], float]:
    """the 2-opt method to generate new schedule

    Args:
        self (object): the algorithm object to fetch necessary params
        schedule (List[int]): the current schedule
        value (float): the current value
        dataLoader (CityDataLoader): where you can query the data from
        maximized (bool, optional): whether to maximize or minimize the target value. Defaults to False.
        initStatus (bool, optional): whether the algorithm is run just started

    Returns:
        Tuple[List[int], float]: new schedule and its value respectively
    """

    def calcDelta(
        schedule: List[int], dataLoader: CityDataLoader, prevPos: int, nextPos: int
    ) -> float:
        """calculate the delta value between schedule before and after

        Args:
            schedule (List[int]): the current schedule
            dataLoader (CityDataLoader): where you can query the data from
            prevPos (int): the 2-opt fore-position
            nextPos (int): the 2-opt post-position

        Returns:
            float: the delta value
        """
        if prevPos == nextPos:
            return 0
        if prevPos + nextPos + 1 == len(schedule):
            return 0

        delta = (
            dataLoader.distMat[
                schedule[(prevPos - 1) % len(schedule)], schedule[nextPos]
            ]
            + dataLoader.distMat[
                schedule[(nextPos + 1) % len(schedule)], schedule[prevPos]
            ]
            - dataLoader.distMat[schedule[prevPos - 1], schedule[prevPos]]
            - dataLoader.distMat[
                schedule[(nextPos + 1) % len(schedule)], schedule[nextPos]
            ]
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

                currDelta = calcDelta(schedule, dataLoader, i, j)

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
        delta = calcDelta(schedule, dataLoader, swapIdx1, swapIdx2)
        delta *= 1 if maximized else -1
        if delta <= 0:
            swapIdx1, swapIdx2 = getIdxViaRandom()
    else:
        raise KeyError(
            f"Methods only include 'random', 'best' and 'mixed'. Method {method} is not existed"
        )

    value += calcDelta(schedule, dataLoader, swapIdx1, swapIdx2)
    for delta in range((swapIdx2 - swapIdx1) // 2 + 1):
        newSchedule[swapIdx1 + delta], newSchedule[swapIdx2 - delta] = (
            newSchedule[swapIdx2 - delta],
            newSchedule[swapIdx1 + delta],
        )

    while not initStatus and self.params.doIntersectAnalysis:
        analysisResult = intersectionAnalysis(newSchedule, dataLoader)
        if analysisResult is None:
            break
        prevIdx, postIdx = analysisResult
        newSchedule, value = intersectionRefactor(
            newSchedule, dataLoader, value, prevIdx, postIdx
        )

    return newSchedule, value


def swap(
    self: object,
    schedule: List[int],
    value: float,
    dataLoader: CityDataLoader,
    maximized: bool = False,
    initStatus: bool = False,
) -> Tuple[List[int], float]:
    """the swap method to generate new schedule

    Args:
        self (object): the algorithm object to fetch necessary params
        schedule (List[int]): the current schedule
        value (float): the current value
        dataLoader (CityDataLoader): where you can query the data from
        maximized (bool, optional): whether to maximize or minimize the target value. Defaults to False.
        initStatus (bool, optional): whether the algorithm is run just started

    Returns:
        Tuple[List[int], float]: new schedule and its value respectively
    """

    def calcSwapDelta(
        schedule: List[int], dataLoader: CityDataLoader, prevIdx: int, postIdx: int
    ) -> float:
        """calculate the delta value between schedule before and after

        Args:
            schedule (List[int]): [description]
            dataLoader (CityDataLoader): [description]
            prevIdx (int): the index for prev-position
            postIdx (int): the index for post-position

        Returns:
            float: the delta value
        """
        prevIdx, postIdx = sorted([prevIdx, postIdx])
        delta = (
            -dataLoader.distMat[schedule[prevIdx], schedule[prevIdx - 1]]
            - dataLoader.distMat[
                schedule[postIdx], schedule[(postIdx + 1) % len(schedule)]
            ]
            + dataLoader.distMat[
                schedule[prevIdx], schedule[(postIdx + 1) % len(schedule)]
            ]
            + dataLoader.distMat[schedule[postIdx], schedule[prevIdx - 1]]
        )

        if prevIdx == 0 and postIdx == len(dataLoader) - 1:
            delta = (
                -dataLoader.distMat[
                    schedule[prevIdx], schedule[(prevIdx + 1) % len(schedule)]
                ]
                - dataLoader.distMat[schedule[postIdx], schedule[postIdx - 1]]
                + dataLoader.distMat[schedule[prevIdx], schedule[postIdx - 1]]
                + dataLoader.distMat[
                    schedule[postIdx], schedule[(prevIdx + 1) % len(schedule)]
                ]
            )
        elif not (abs(prevIdx - postIdx) <= 1):
            delta += (
                -dataLoader.distMat[
                    schedule[prevIdx], schedule[(prevIdx + 1) % len(schedule)]
                ]
                - dataLoader.distMat[schedule[postIdx], schedule[postIdx - 1]]
                + dataLoader.distMat[schedule[prevIdx], schedule[postIdx - 1]]
                + dataLoader.distMat[
                    schedule[postIdx], schedule[(prevIdx + 1) % len(schedule)]
                ]
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
        prevIdx, postIdx = analysisResult
        schedule, value = intersectionRefactor(
            schedule, dataLoader, value, prevIdx, postIdx
        )

    return schedule, value


def crossover(
    self: object,
    chroms: List[Tuple[List[int], float]],
    calcValue: Callable,
    dataLoader: CityDataLoader,
) -> Tuple[List[int], float]:
    """[summary]

    Args:
        self (object): [description]
        chroms (List[Tuple[List[int], float]]): [description]
        calcValue (Callable): [description]
        dataLoader (CityDataLoader): [description]

    Returns:
        Tuple[List[int], float]: [description]
    """
    # init params
    chroms.sort(key=lambda x: x[-1], reverse=self.params.maximize)
    fitnesses = [
        value if self.params.maximize else 1 / value for schedule, value in chroms
    ]

    # choose two chromosomes to crossover
    chosenChromo = []
    while len(chosenChromo) < 2:
        probPtr = random.random() * sum(fitnesses)
        for idx, fitness in enumerate(fitnesses):
            if fitness > probPtr and idx not in chosenChromo:
                chosenChromo.append(idx)
                break
            probPtr -= fitness

    ###########################################################
    # generate & return new schedule and its value respectively
    ###########################################################
    tmp = {}
    schedule = [None for _ in chroms[chosenChromo[0]][0]]

    startIdx = random.randint(0, len(chroms[0][0]) - 2)
    endIdx = random.randint(startIdx + 1, len(chroms[0][0]) - 1)

    # check and pair the cityId which is not include in the other sequence [startIdx -> endIdx]
    firstVacancy = []
    secondVacancy = []
    for idx in range(startIdx, endIdx + 1):
        firstCityId = chroms[chosenChromo[0]][0][idx]
        if firstCityId not in chroms[chosenChromo[1]][0][startIdx : endIdx + 1]:
            firstVacancy.append(firstCityId)

        secondCityId = chroms[chosenChromo[1]][0][idx]
        if secondCityId not in chroms[chosenChromo[0]][0][startIdx : endIdx + 1]:
            secondVacancy.append(secondCityId)

    for key, value in zip(secondVacancy, firstVacancy):
        tmp[key] = value

    # generate new schedule and its value

    for idx in range(startIdx, endIdx + 1):
        schedule[idx] = chroms[chosenChromo[1]][0][idx]

    for idx in range(len(chroms[0][0])):
        if startIdx <= idx <= endIdx:
            continue

        cityId = chroms[chosenChromo[0]][0][idx]
        if cityId in schedule:
            schedule[idx] = tmp[chroms[chosenChromo[0]][0][idx]]
        else:
            schedule[idx] = chroms[chosenChromo[0]][0][idx]

    value = calcValue(schedule, dataLoader)

    return schedule, value


def mutate(
    self: object,
    schedule: List[int],
    value: float,
    dataLoader: CityDataLoader,
) -> Tuple[List[int], float]:
    """[summary]

    Args:
        self (object): [description]
        schedule (List[int]): [description]
        value (float): [description]
        dataLoader (CityDataLoader): [description]

    Returns:
        Tuple[List[int], float]: [description]
    """

    def calcDelta(
        schedule: List[int], dataLoader: CityDataLoader, prevIdx: int, postIdx: int
    ) -> float:
        """calculate the delta value between schedule before and after

        Args:
            schedule (List[int]): [description]
            dataLoader (CityDataLoader): [description]
            prevIdx (int): the index for prev-position
            postIdx (int): the index for post-position

        Returns:
            float: the delta value
        """
        prevIdx, postIdx = sorted([prevIdx, postIdx])
        delta = (
            -dataLoader.distMat[schedule[prevIdx], schedule[prevIdx - 1]]
            - dataLoader.distMat[
                schedule[postIdx], schedule[(postIdx + 1) % len(schedule)]
            ]
            + dataLoader.distMat[
                schedule[prevIdx], schedule[(postIdx + 1) % len(schedule)]
            ]
            + dataLoader.distMat[schedule[postIdx], schedule[prevIdx - 1]]
        )

        if prevIdx == 0 and postIdx == len(dataLoader) - 1:
            delta = (
                -dataLoader.distMat[
                    schedule[prevIdx], schedule[(prevIdx + 1) % len(schedule)]
                ]
                - dataLoader.distMat[schedule[postIdx], schedule[postIdx - 1]]
                + dataLoader.distMat[schedule[prevIdx], schedule[postIdx - 1]]
                + dataLoader.distMat[
                    schedule[postIdx], schedule[(prevIdx + 1) % len(schedule)]
                ]
            )
        elif not (abs(prevIdx - postIdx) <= 1):
            delta += (
                -dataLoader.distMat[
                    schedule[prevIdx], schedule[(prevIdx + 1) % len(schedule)]
                ]
                - dataLoader.distMat[schedule[postIdx], schedule[postIdx - 1]]
                + dataLoader.distMat[schedule[prevIdx], schedule[postIdx - 1]]
                + dataLoader.distMat[
                    schedule[postIdx], schedule[(prevIdx + 1) % len(schedule)]
                ]
            )

        return delta

    if random.random() < self.params.mutateProb:
        prevIdx = random.randint(0, len(schedule) - 2)
        postIdx = random.randint(prevIdx + 1, len(schedule) - 1)
        value += calcDelta(schedule, dataLoader, prevIdx, postIdx)
        schedule[prevIdx], schedule[postIdx] = schedule[prevIdx], schedule[postIdx]

    return schedule, value
