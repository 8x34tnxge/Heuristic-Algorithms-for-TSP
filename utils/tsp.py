import random

import numpy as np


def initSolution(dataLoader, distFunc):
    schedule = []
    for cityId, cityGeoInfo in dataLoader:
        schedule.append(cityId)
    random.shuffle(schedule)

    value = distFunc(schedule, dataLoader)

    return schedule, value

def twoOpt(self, schedule, value, dataLoader, maximized=False):
    def calcDelta(schedule, prevPos, nextPos):
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

    def getIdxViaRandom():
        randomIdx1 = random.randint(0, len(schedule) - 1)
        randomIdx2 = random.randint(0, len(schedule) - 1)
        while randomIdx1 == randomIdx2 or randomIdx1 + randomIdx2 + 1 == len(schedule):
            randomIdx2 = random.randint(0, len(schedule) - 1)
        randomIdx1, randomIdx2 = sorted([randomIdx1, randomIdx2])
        return randomIdx1, randomIdx2

    def getIdxViaBest():
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

    method = self.params.method
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


def distFunc(schedule, data):
    dist = 0
    for i in range(len(schedule)):
        prevGeoInfo = data[schedule[i]]
        nextGeoInfo = data[schedule[(i + 1) % len(schedule)]]

        deltaX = prevGeoInfo.x - nextGeoInfo.x
        deltaY = prevGeoInfo.y - nextGeoInfo.y
        dist += np.linalg.norm(np.array([deltaX, deltaY]))
    return dist
