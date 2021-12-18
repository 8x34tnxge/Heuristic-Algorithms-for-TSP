import random
from typing import Callable, List

import numpy as np
from tqdm import tqdm
from utils.base import DataLoader
from utils.tsp import distFunc, initSolution, twoOpt

from alglib.base import Base
from alglib.param import AdaptiveSA_Param, SA_Param


class SimulatedAnnealing(Base):
    """
    Simulated Annealing Algorithm
    """

    def __init__(self, params: SA_Param) -> None:
        """initialize SA algorithm

        Args:
            params (SA_Param): the parameter object for Simulated Annealing algorithm
        """
        self.bestValueWatcher = []
        self.params: SA_Param = params
        super().__init__()

    def __setattr__(self, name: str, value: float) -> None:
        """the method to set value watcher attribute

        Args:
            name (str): watcher's name
            value (float): the notable value needed to be recorded
        """
        if name == "bestValue":
            self.bestValueWatcher.append(value)
        super().__setattr__(name, value)

    def run(
        self,
        dataLoader: DataLoader,
        initSchedule: Callable = initSolution,
        calcValue: Callable = distFunc,
        fetchNewSchedule: Callable = twoOpt,
    ) -> None:
        """the main procedure for SA

        Args:
            dataLoader (DataLoader): where you can query data from
            initSchedule (Callable, optional): the method to init schedule. Defaults to initSolution.
            calcValue (Callable, optional): the method to calc value from given schedule. Defaults to distFunc.
            fetchNewSchedule (Callable, optional): the method to fetch new schedule. Defaults to twoOpt.

        Raises:
            ValueError: check whether the cool rate is in [0, 1)
        """
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

        # define the util functions
        def updateLocalSchedule(
            schedule: List, value: float, temperate: float, maximize=False
        ) -> None:
            """the method to update local schedule

            Args:
                schedule (List): the current schedule
                value (float): the current value
                temperate (float): the current temperate
                maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
            """
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

        def updateGlobalSchedule(schedule, value, maximize=False) -> None:
            """the method to update global schedule

            Args:
                schedule (List): the current schedule
                value (float): the current value
                maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
            """
            if not maximize and value >= self.bestValue:
                return
            if maximize and value <= self.bestValue:
                return
            self.bestSchedule = []
            self.bestSchedule.extend(schedule)
            self.bestValue = value

        # main procedure
        schedule, value = initSchedule(dataLoader, calcValue)

        updateLocalSchedule(schedule, value, temperate, maximize=self.params.maximize)
        updateGlobalSchedule(schedule, value, maximize=self.params.maximize)

        while temperate > terminatedTemperate:
            tbar = tqdm(range(epochNum))
            tbar.set_description_str(f"Current Temperate: {temperate:.6f}")
            for epoch in tbar:
                schedule, value = fetchNewSchedule(
                    self,
                    self.localSchedule,
                    self.localValue,
                    dataLoader,
                    initStatus=(epoch / epochNum) < self.params.initStatusJudgement,
                )

                updateLocalSchedule(
                    schedule, value, temperate, maximize=self.params.maximize
                )
                updateGlobalSchedule(schedule, value, maximize=self.params.maximize)

            temperate *= coolRate


class AdaptiveSimulatedAnnealing(SimulatedAnnealing):
    """
    Adaptive Simulated Annealing Algorithm
    Reference:
    Wu, Guohua, et al. "Satellite observation scheduling with a novel adaptive simulated annealing algorithm and a dynamic task clustering strategy." Computers & Industrial Engineering 113 (2017): 576-588.
    """

    def __init__(self, params: AdaptiveSA_Param) -> None:
        """the method initialize the Adaptive SA algorithm

        Args:
            params (AdaptiveSA_Param): the parameter object for the Adaptive Simulated Annealing algorithm
        """
        super().__init__(params)
        self.cnt = 0

    def run(
        self,
        dataLoader: DataLoader,
        initSchedule: Callable = initSolution,
        calcValue: Callable = distFunc,
        fetchNewSchedule: Callable = twoOpt,
    ) -> None:
        """the main procedure for Adaptive SA

        Args:
            dataLoader (DataLoader): where you can query data from
            initSchedule (Callable, optional): the method to init schedule. Defaults to initSolution.
            calcValue (Callable, optional): the method to calc value from given schedule. Defaults to distFunc.
            fetchNewSchedule (Callable, optional): the method to fetch new schedule. Defaults to twoOpt.
        """
        # load params
        self.cnt = 0
        minTemperate = self.params.minTemperate
        penalWeight = self.params.penalWeight
        delta = self.params.delta
        epochNum = self.params.epochNum

        # define the util functions
        def updateLocalSchedule(schedule, value, temperate, maximize=False):
            """the method to update local schedule

            Args:
                schedule (List): the current schedule
                value (float): the current value
                temperate (float): the current temperate
                maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
            """
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
            """the method to update global schedule

            Args:
                schedule (List): the current schedule
                value (float): the current value
                maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
            """
            if not maximize and value > self.bestValue:
                return
            if maximize and value < self.bestValue:
                return
            self.bestSchedule = []
            self.bestSchedule.extend(schedule)
            self.bestValue = value

        # init the first route or schedule
        schedule, value = initSchedule(dataLoader, calcValue)

        updateLocalSchedule(
            schedule, value, minTemperate, maximize=self.params.maximize
        )
        updateGlobalSchedule(schedule, value, maximize=self.params.maximize)

        # solve
        tbar = tqdm(range(epochNum))
        for epoch in tbar:
            temperate = minTemperate + penalWeight * np.log(1 + self.cnt / delta)
            tbar.set_description_str(f"Current Temperate: {temperate:.6f}")

            schedule, value = fetchNewSchedule(
                self,
                self.localSchedule,
                self.localValue,
                dataLoader,
                initStatus=(epoch / epochNum) < self.params.initStatusJudgement,
            )

            updateLocalSchedule(
                schedule, value, temperate, maximize=self.params.maximize
            )
            updateGlobalSchedule(schedule, value, maximize=self.params.maximize)
