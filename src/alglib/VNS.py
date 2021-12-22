from typing import Callable, List

from tqdm import tqdm
from utils.base import DataLoader
from utils.tsp import distFunc, initSolution

from alglib.base import Base
from alglib.param import VNS_Param


class VariableNeighborhoodSearch(Base):
    """
    Variable Neighborhood Search Algorithm
    """

    def __init__(self, params: VNS_Param) -> None:
        """initialize VNS algorithm

        Args:
            params (VNS_Param): the parameter object for Variable Neighborhood Search algorithm
        """
        self.bestValueWatcher = []
        self.params: VNS_Param = params
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
    ) -> None:
        """the main procedure for VNS

        Args:
            dataLoader (DataLoader): where you can query data from
            initSchedule (Callable, optional): the method to init schedule. Defaults to initSolution.
            calcValue (Callable, optional): the method to calc value from given schedule. Defaults to distFunc.
            fetchNewSchedule (Callable, optional): the method to fetch new schedule. Defaults to twoOpt.
        """
        # load params
        epochNum = self.params.epochNum

        # define the util functions
        def updateLocalSchedule(
            schedule: List[int],
            value: float,
            maximize: bool = False,
            force: bool = False,
        ) -> None:
            """the method to update local schedule

            Args:
                schedule (List[int]): the current schedule
                value (float): the current value
                maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
                force (bool, optional): whether to force localSchedule and its value to update. Defaults to False.
            """
            if not force and (self.localValue < value) ^ maximize:
                return

            self.localSchedule = []
            self.localSchedule.extend(schedule)
            self.localValue = value

        def updateGlobalSchedule(
            schedule: List[int], value: float, maximize: bool = False
        ) -> None:
            """the method to update global schedule

            Args:
                schedule (List[int]): the current schedule
                value (float): the current value
                maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
            """
            if (self.bestValue < value) ^ maximize:
                return

            self.bestSchedule = []
            self.bestSchedule.extend(schedule)
            self.bestValue = value

        def updateCounter(counter: int, value: float) -> int:
            """check whether local value is updated

            Args:
                counter (int): the solver counter
                value (float): the current value

            Returns:
                int: new solver counter
            """
            if value == self.localValue:
                return counter + 1
            else:
                return 0

        # main procedure
        for epoch in tqdm(range(epochNum)):
            schedule, value = initSchedule(dataLoader, calcValue)

            updateLocalSchedule(
                schedule, value, force=True, maximize=self.params.maximize
            )
            updateGlobalSchedule(schedule, value, maximize=self.params.maximize)

            methodCnt = 0
            while methodCnt < len(self.params.methods):
                fetchNewSchedule = self.params.methods[methodCnt]
                # print(self.localValue)
                schedule, value = fetchNewSchedule(
                    self,
                    self.localSchedule,
                    self.localValue,
                    dataLoader,
                    initStatus=(epoch / epochNum) < self.params.initStatusJudgement,
                )
                # print(value)
                # print(distFunc(schedule, dataLoader))
                # print()
                updateLocalSchedule(schedule, value, maximize=self.params.maximize)
                updateGlobalSchedule(schedule, value, maximize=self.params.maximize)
                methodCnt = updateCounter(methodCnt, value)
