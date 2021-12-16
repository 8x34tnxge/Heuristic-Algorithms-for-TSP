import random
from typing import Any, NoReturn

import numpy as np
from tqdm import tqdm
from utils.dataloader import CityDataLoader
from utils.tsp import distFunc, initSolution, twoOpt

from algCore.base import Base
from algCore.param import SA_Param


class SimulatedAnnealing(Base):
    def __init__(self, params: SA_Param) -> NoReturn:
        self.bestValueWatcher = []
        self.params: SA_Param = params
        super().__init__()

    def __setattr__(self, name, value):
        if name == 'bestValue':
            self.bestValueWatcher.append(value)
        super().__setattr__(name, value)

    def run(
        self,
        dataLoader: CityDataLoader,
        initSolution=initSolution,
        distFunc=distFunc,
        fetchNewSchedule=twoOpt,
    ) -> NoReturn:
        # define the util functions
        def updateLocalSchedule(schedule, value, temperate, maximize=False):
            isDenied = False
            if (not maximize and value > self.localValue) or (
                maximize and value < self.localvalue
            ):
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
            if not maximize and value >= self.bestValue:
                return
            if maximize and value <= self.bestValue:
                return
            self.bestSchedule = []
            self.bestSchedule.extend(schedule)
            self.bestValue = value

        # load params
        temperate = self.params.initialTemperate
        terminatedTemperate = self.params.terminatedTemperate
        coolRate = self.params.coolRate
        epochNum = self.params.epochNum

        try:
            assert 0 <= coolRate < 1
        except AssertionError:
            raise ValueError(
                f"CoolRate must be in [0, 1). Current coolRate's value is {coolRate}"
            )

        # main procedure
        schedule, value = initSolution(dataLoader, distFunc)

        updateLocalSchedule(schedule, value, temperate)
        updateGlobalSchedule(schedule, value)

        while temperate > terminatedTemperate:
            tbar = tqdm(range(epochNum))
            tbar.set_description_str(f"Current Temperate: {temperate:.6f}")
            for epoch in tbar:
                schedule, value = fetchNewSchedule(
                    self, self.localSchedule, self.localValue, dataLoader
                )

                updateLocalSchedule(schedule, value, temperate)
                updateGlobalSchedule(schedule, value)

            temperate *= coolRate


class AdaptiveSimulatedAnnealing(SimulatedAnnealing):
    def __init__(self, params):
        super().__init__(params)
        self.cnt = 0

    def run(
        self,
        dataLoader: CityDataLoader,
        initSolution=initSolution,
        distFunc=distFunc,
        fetchNewSchedule=twoOpt,
    ) -> NoReturn:
        # define the util functions
        def updateLocalSchedule(schedule, value, temperate, maximize=False):
            isDenied = False
            if (not maximize and value > self.localValue) or (
                maximize and value < self.localvalue
            ):
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

        # load params
        self.cnt = 0
        minTemperate = self.params.minTemperate
        penalWeight = self.params.penalWeight
        delta = self.params.delta
        epochNum = self.params.epochNum

        # init the first route or schedule
        schedule, value = initSolution(dataLoader, distFunc)

        updateLocalSchedule(schedule, value, minTemperate)
        updateGlobalSchedule(schedule, value)

        # solve
        tbar = tqdm(range(epochNum))
        for epoch in tbar:
            temperate = minTemperate + penalWeight * np.log(1 + self.cnt / delta)
            tbar.set_description_str(f"Current Temperate: {temperate:.6f}")

            schedule, value = fetchNewSchedule(
                self, self.localSchedule, self.localValue, dataLoader
            )

            updateLocalSchedule(schedule, value, temperate)
            updateGlobalSchedule(schedule, value)
