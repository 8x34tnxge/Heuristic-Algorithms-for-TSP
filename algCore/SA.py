from algCore.base import Base
from tqdm import tqdm
import random
from algCore.param import SA_Param
from typing import NoReturn, Any
import numpy as np
from random import shuffle
from utils.dataloader import CityDataLoader

class SimulatedAnnealing(Base):
    def __init__(self, params: SA_Param) -> NoReturn:
        self.bestValueWatcher = []
        self.params: SA_Param = params
        super().__init__()

    def __setattr__(self, name, value):
        if name == 'bestValue':
            self.bestValueWatcher.append(value)
        super().__setattr__(name, value)

    def run(self, dataLoader: CityDataLoader, distFunc=None) -> NoReturn:
        # define the util functions
        def updateLocalSchedule(schedule, value, temperate, maximize=False):
            isDenied = False
            if (not maximize and value > self.localValue) or \
            (maximize and value < self.localvalue):
                isDenied = True
                deltaValue = self.localValue - value
                if maximize:
                    deltaValue *= -1
                compulsoryProb = np.exp2(deltaValue / temperate)
                prob = random.random()
            if isDenied and prob > compulsoryProb:
                return
            self.localSchedule = []
            self.localSchedule.extend(schedule)
            self.localValue = value

        def updateGlobalSchedule(schedule, value, maximize=False):
            if not maximize and value > self.bestValue:
                return
            if maximize and value < self.bestValue:
                return
            self.bestSchedule = []
            self.bestSchedule.extend(schedule)
            self.bestValue = value

        # load params
        temperate = self.params.initialTemperate
        terminatedTemperate = self.params.terminatedTemperate
        coolRate = self.params.coolRate
        epochNum = self.params.epochNum
        if distFunc is None:
            distFunc = self.distFunc

        try:
            assert 0 <= coolRate < 1
        except AssertionError:
            raise ValueError(f"CoolRate must be in [0, 1). Current coolRate's value is {coolRate}")

        # init the first route or schedule
        schedule = []
        for cityId, cityGeoInfo in dataLoader:
            schedule.append(cityId)
        shuffle(schedule)
        value = self.distFunc(schedule, dataLoader)

        updateLocalSchedule(schedule, value, temperate)
        updateGlobalSchedule(schedule, value)

        # solve
        while temperate > terminatedTemperate:
            tbar = tqdm(range(epochNum))
            tbar.set_description_str(f"Current Temperate: {temperate:.6f}")
            for epoch in tbar:
                schedule, value = self.twoOpt(self.localSchedule, value, dataLoader, distFunc)
                value = distFunc(schedule, dataLoader)

                updateLocalSchedule(schedule, value, temperate)
                updateGlobalSchedule(schedule, value)

            temperate *= coolRate
    
    def twoOpt(self, schedule, value, dataLoader, distFunc, maximized=False):
        def calcDelta(schedule, prevPos, nextPos):
            if prevPos == nextPos:
                return 0
            if prevPos + nextPos + 1 == len(schedule):
                return 0

            prevGeoInfo4i = dataLoader[schedule[(prevPos + len(schedule) - 1) % len(schedule)]]
            currGeoInfo4i = dataLoader[schedule[prevPos]]

            currGeoInfo4j = dataLoader[schedule[nextPos]]
            nextGeoInfo4j = dataLoader[schedule[(nextPos + len(schedule) + 1) % len(schedule)]]

            delta = np.linalg.norm(np.array([
                prevGeoInfo4i.x - currGeoInfo4j.x,
                prevGeoInfo4i.y - currGeoInfo4j.y
            ])) + np.linalg.norm(np.array([
                nextGeoInfo4j.x - currGeoInfo4i.x,
                nextGeoInfo4j.y - currGeoInfo4i.y
            ])) - np.linalg.norm(np.array([
                prevGeoInfo4i.x - currGeoInfo4i.x,
                prevGeoInfo4i.y - currGeoInfo4i.y
            ])) - np.linalg.norm(np.array([
                nextGeoInfo4j.x - currGeoInfo4j.x,
                nextGeoInfo4j.y - currGeoInfo4j.y
            ]))
            return delta

        def getIdxViaRandom():
            randomIdx1 = random.randint(0, len(schedule)-1)
            randomIdx2 = random.randint(0, len(schedule)-1)
            while randomIdx1 == randomIdx2 or randomIdx1 + randomIdx2 + 1 == len(schedule):
                randomIdx2 = random.randint(0, len(schedule)-1)
            randomIdx1, randomIdx2 = sorted([randomIdx1, randomIdx2])
            return randomIdx1, randomIdx2

        def getIdxViaBest():
            delta = -np.inf
            bestIdx1, bestIdx2 = -1, -1
            for i in range(len(schedule)-1):
                for j in range(i+1, len(schedule)):
                    if j - i + 1 == len(schedule) - 1:
                        continue
                    
                    if i + j + 1 == len(schedule):
                        continue

                    currDelta = calcDelta(schedule, i, j)

                    currDelta *= 1 if maximized else -1
                    if currDelta > delta:
                        delta = currDelta
                        bestIdx1, bestIdx2= i, j
                    
            return bestIdx1, bestIdx2

        method = self.params.method
        newSchedule = []
        newSchedule.extend(schedule)
        if method == 'random':
            swapIdx1, swapIdx2 = getIdxViaRandom()
        elif method == 'best':
            swapIdx1, swapIdx2 = getIdxViaBest()
        elif method == 'mixed':
            swapIdx1, swapIdx2 = getIdxViaBest()
            if calcDelta(schedule, swapIdx1, swapIdx2) <= 0:
                swapIdx1, swapIdx2 = getIdxViaRandom()
        else:
            raise KeyError(f"Methods only include 'random', 'best' and 'mixed'. Method {method} is not existed")

        for delta in range((swapIdx2 - swapIdx1) // 2 + 1):
            newSchedule[swapIdx1+delta], newSchedule[swapIdx2-delta] = newSchedule[swapIdx2-delta], newSchedule[swapIdx1+delta]
        value += calcDelta(schedule, swapIdx1, swapIdx2) * (1 if maximized else -1)

        return newSchedule, value
    
    def distFunc(self, schedule, data):
        dist = 0
        for i in range(len(schedule)):
            prevGeoInfo = data[schedule[i]]
            nextGeoInfo = data[schedule[(i+1)%len(schedule)]]

            deltaX = prevGeoInfo.x - nextGeoInfo.x
            deltaY = prevGeoInfo.y - nextGeoInfo.y
            dist += np.linalg.norm(np.array([deltaX, deltaY]))
        return dist


class AdaptiveSimulatedAnnealing(SimulatedAnnealing):
    def __init__(self, params):
        super().__init__(params)
        self.cnt = 0

    def run(self, dataLoader: CityDataLoader, distFunc=None) -> NoReturn:
        # load params
        self.cnt = 0
        minTemperate = self.params.minTemperate
        penalWeight = self.params.penalWeight
        delta = self.params.delta
        epochNum = self.params.epochNum
        if distFunc is None:
            distFunc = self.distFunc

        # define the util functions
        def updateLocalSchedule(schedule, value, temperate, maximize=False):
            isDenied = False
            if (not maximize and value > self.localValue) or \
            (maximize and value < self.localvalue):
                isDenied = True
                deltaValue = self.localValue - value
                if maximize:
                    deltaValue *= -1
                compulsoryProb = np.exp2(deltaValue / temperate)
                prob = random.random()
            if isDenied and prob > compulsoryProb:
                self.cnt += 1
                return
            self.cnt = 0
            self.localSchedule = []
            self.localSchedule.extend(schedule)
            self.localValue = value

        def updateGlobalSchedule(schedule, value, maximize=False):
            if not maximize and value > self.bestValue:
                return
            if maximize and value < self.bestValue:
                return
            self.bestSchedule = []
            self.bestSchedule.extend(schedule)
            self.bestValue = value


        # init the first route or schedule
        schedule = []
        for cityId, cityGeoInfo in dataLoader:
            schedule.append(cityId)
        shuffle(schedule)
        value = self.distFunc(schedule, dataLoader)

        updateLocalSchedule(schedule, value, minTemperate)
        updateGlobalSchedule(schedule, value)

        # solve
        tbar = tqdm(range(epochNum))
        for epoch in tbar:
            temperate = minTemperate + penalWeight * np.log(1 + self.cnt / delta) 
            tbar.set_description_str(f"Current Temperate: {temperate:.6f}")
            
            schedule, value = self.twoOpt(self.localSchedule, value, dataLoader, distFunc)
            value = distFunc(schedule, dataLoader)

            updateLocalSchedule(schedule, value, temperate)
            updateGlobalSchedule(schedule, value)
