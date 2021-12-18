from collections import namedtuple

import numpy as np
from yacs.config import CfgNode

from utils.base import DataLoader


class CityDataLoader(DataLoader):
    def __init__(self, data: CfgNode) -> None:
        """the data loader containing the city data

        Args:
            data (CfgNode): the city data
        """
        coordPair = namedtuple("coordPair", ["x", "y"])
        processedData = dict()
        for elem in data:
            id = elem["id"]
            x = elem["cityCoord"][0]["x"]
            y = elem["cityCoord"][1]["y"]
            processedData[id] = coordPair(x, y)

        super().__init__(processedData)

        self.distMat = np.ones((len(data) + 1, len(data) + 1))
        for col, _ in self.data.items():
            for row, _ in self.data.items():
                if col > row:
                    continue

                elif col == row:
                    self.distMat[row][col] = 0
                    continue

                self.distMat[row][col] = np.linalg.norm(
                    np.array([self[col].x - self[row].x, self[col].y - self[row].y])
                )
                self.distMat[col][row] = np.linalg.norm(
                    np.array([self[col].x - self[row].x, self[col].y - self[row].y])
                )
        # self.distMat = np.array(
        #     [
        #         [
        #             np.linalg.norm(
        #                 np.array([self[col].x - self[row].x, self[col].y - self[row].y])
        #             )
        #             for col, _ in self.data.items()
        #         ]
        #         for row, _ in self.data.items()
        #     ]
        # )

