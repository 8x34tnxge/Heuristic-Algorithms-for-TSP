from collections import namedtuple

from yacs.config import CfgNode

from utils.base import DataLoader


class CityDataLoader(DataLoader):
    def __init__(self, data: CfgNode) -> None:
        """the data loader containing the city data

        Args:
            data (CfgNode): the city data
        """
        coordPair = namedtuple('coordPair', ['x', 'y'])
        processedData = dict()
        for elem in data:
            id = elem['id']
            x = elem['cityCoord'][0]['x']
            y = elem['cityCoord'][1]['y']
            processedData[id] = coordPair(x, y)

        super().__init__(processedData)
    