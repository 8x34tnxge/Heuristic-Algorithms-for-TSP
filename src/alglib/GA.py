from typing import Callable, List

from tqdm import tqdm
from utils.base import DataLoader
from utils.tsp import crossover, distFunc, initSolution, mutate

from alglib.base import Base
from alglib.param import GA_Param


class GeneticAlgorithm(Base):
    """
    Genetic Algorithm
    """

    def __init__(self, params: GA_Param) -> None:
        """initialize GA algorithm

        Args:
            params (GA_Param): the parameter object for Genetic Algorithm
        """
        self.bestValueWatcher = []
        self.params: GA_Param = params
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
        crossover: Callable = crossover,
        mutate: Callable = mutate,
    ) -> None:
        """the main procedure for GA

        Args:
            dataLoader (DataLoader): where you can query data from
            initSchedule (Callable, optional): the method to init schedule. Defaults to initSolution.
            calcValue (Callable, optional): the method to calc value from given schedule. Defaults to distFunc.
            crossover (Callable, optional): the method to let chromosomes crossover. Defaults to crossover.
            mutate (Callable, optional): the method to let chromosomes mutate. Defaults to mutate.
        """
        # load params
        epochNum = self.params.epochNum
        popNum = self.params.popNum
        newPopRate = self.params.newPopRate
        maximize = self.params.maximize

        # define the util functions
        def updateLocalSchedule(schedule, value, maximize: bool = False) -> None:
            """the method to update particle

            Args:
                maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
            """
            if (self.localValue < value) ^ maximize:
                return

            self.localSchedule = []
            self.localSchedule.extend(schedule)
            self.localValue = value

        def updateGlobalSchedule(schedule, value, maximize: bool = False) -> None:
            """the method to update global schedule

            Args:
                maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
            """
            if (self.bestValue < value) ^ maximize:
                return

            self.bestSchedule = []
            self.bestSchedule.extend(schedule)
            self.bestValue = value

        # main procedure
        chroms = []
        for _ in range(popNum):
            schedule, value =  initSchedule(dataLoader, calcValue)
            chroms.append(
                (schedule, value)
            )
            updateLocalSchedule(schedule, value, maximize=maximize)
        updateGlobalSchedule(self.localSchedule, self.localValue, maximize=maximize)

        for epoch in range(epochNum):
            newChroms = []
            newPopNum = int(popNum * newPopRate)

            while len(newChroms) < newPopNum:
                schedule, value = crossover(self, chroms, calcValue, dataLoader)
                schedule, value = mutate(self, schedule, value, dataLoader)
                newChroms.append(
                    (schedule, value)
                )

            chroms.sort(key=lambda x: x[-1], reverse=maximize)
            newChroms.sort(key=lambda x: x[-1], reverse=maximize)

            while len(chroms) + len(newChroms) > popNum and (newChroms[-1][-1] > chroms[-1][-1]) ^ maximize:
                chroms.pop()
            chroms.extend(newChroms)

            for schedule, value in chroms:
                updateLocalSchedule(schedule, value, maximize)
            updateGlobalSchedule(self.localSchedule, self.localValue, maximize)