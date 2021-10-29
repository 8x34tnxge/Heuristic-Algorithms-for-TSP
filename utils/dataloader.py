from yacs.config import CfgNode
from utils.base import DataLoader
from collections import namedtuple


class CityDataLoader(DataLoader):
    def __init__(self, data: CfgNode) -> None:
        coordPair = namedtuple('coordPair', ['x', 'y'])
        processedData = dict()
        for elem in data:
            id = elem['id']
            x = elem['cityCoord'][0]['x']
            y = elem['cityCoord'][1]['y']
            processedData[id] = coordPair(x, y)

        super().__init__(processedData)
    