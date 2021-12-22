from typing import Callable, List

import numpy as np
from tqdm import tqdm
from utils.base import DataLoader
from utils.tsp import distFunc, initSolution, swap

from alglib.base import Base
from alglib.param import PSO_Param


class ParticleSwarmOptimization(Base):
    """
    Particle Swarm Optimization Algorithm
    """

    def __init__(self, params: PSO_Param) -> None:
        """initialize GA algorithm

        Args:
            params (PSO_Param): the parameter object for Particle Swarm Optimization algorithm
        """
        self.bestValueWatcher = []
        self.params: PSO_Param = params
        self.particles = [-1 for _ in range(self.params.particleNum)]
        self.particleValues = [np.inf for _ in range(self.params.particleNum)]
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
        fetchNewSchedule: Callable = swap,
    ) -> None:
        """the main procedure for PSO

        Args:
            dataLoader (DataLoader): where you can query data from
            initSchedule (Callable, optional): the method to init schedule. Defaults to initSolution.
            calcValue (Callable, optional): the method to calc value from given schedule. Defaults to distFunc.
            fetchNewSchedule (Callable, optional): the method to fetch new schedule. Defaults to twoOpt.

        Raises:
            ValueError: check whether the cool rate is in [0, 1)
        """
        # load params
        epochNum = self.params.epochNum

        # define the util functions
        def updateParticle(
            particleIdx: int,
            schedule: List[int],
            value: float,
            maximize: bool = False,
            force: bool = False,
        ) -> None:
            """the method to update particle

            Args:
                particleIdx (int): the index of the current particle
                schedule (List[int]): the current schedule
                value (float): the current value
                maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
                force (bool, optional): whether to force localSchedule and its value to update. Defaults to False.
            """
            if not force and (self.particleValues[particleIdx] > value) ^ maximize:
                return

            self.particles[particleIdx] = []
            self.particles[particleIdx].extend(schedule)
            self.particleValues[particleIdx] = value

        def updateLocalSchedule(maximize: bool = False,) -> None:
            """the method to update particle

            Args:
                maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
            """
            for schedule, value in zip(self.particles, self.particleValues):
                if (self.localValue < value) ^ maximize:
                    return

                self.localSchedule = []
                self.localSchedule.extend(schedule)
                self.localValue = value

        def updateGlobalSchedule(maximize: bool = False) -> None:
            """the method to update global schedule

            Args:
                maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
            """
            for schedule, value in zip(self.particles, self.particleValues):
                if (self.bestValue < value) ^ maximize:
                    return

                self.bestSchedule = []
                self.bestSchedule.extend(schedule)
                self.bestValue = value

        # main procedure
        for particleIdx in range(self.params.particleNum):
            schedule, value = initSchedule(dataLoader, calcValue)
            updateParticle(
                particleIdx, schedule, value, maximize=self.params.maximize, force=True
            )

        updateLocalSchedule(maximize=self.params.maximize)
        updateGlobalSchedule(maximize=self.params.maximize)

        for epoch in tqdm(range(epochNum)):
            for idx, (particle, particleValue) in enumerate(
                zip(self.particles, self.particleValues)
            ):
                schedule, value = fetchNewSchedule(
                    self,
                    particle,
                    particleValue,
                    dataLoader,
                    initStatus=(epoch / epochNum) < self.params.initStatusJudgement,
                )
                updateParticle(
                    idx, schedule, value, maximize=self.params.maximize, force=True
                )
            updateLocalSchedule(maximize=self.params.maximize)
            updateGlobalSchedule(maximize=self.params.maximize)
