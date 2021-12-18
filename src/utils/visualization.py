import os
from typing import List

import matplotlib.pyplot as plt

from utils.base import DataLoader


def visualize(
    schedule: List[int],
    dataLoader: DataLoader,
    valueWatcher: List[float],
    fileDir: str,
    fileName: str,
    save=False,
):
    """the visualization method to visualize the algorithm result

    Args:
        schedule (List[int]): the final schedule
        dataLoader (DataLoader): where you can query the data from
        valueWatcher (List[float]): the value record
        fileDir (str): output directory
        fileName (str): output file name
        save (bool, optional): whether to save the file to fileDir. Defaults to False.
    """
    validValue = valueWatcher[1:]
    bestValue = valueWatcher[-1]
    fig = plt.figure()

    plt.subplot(2, 1, 1)
    plt.title(f"best value: {bestValue:.6f}")
    plt.plot(validValue)

    plt.subplot(2, 1, 2)
    plt.title("route")
    plt.xlabel("")
    plt.ylabel("")
    x = []
    y = []
    for idInfo, geoInfo in dataLoader:
        x.append(geoInfo.x)
        y.append(geoInfo.y)
    plt.scatter(x, y)
    schedule.append(schedule[0])
    for prev in range(len(schedule) - 1):
        next = prev + 1
        plt.plot(
            [dataLoader[schedule[prev]].x, dataLoader[schedule[next]].x],
            [dataLoader[schedule[prev]].y, dataLoader[schedule[next]].y],
        )

    if not os.path.exists(fileDir):
        os.mkdir(fileDir)

    if save:
        fig.savefig(f"{fileDir}/{fileName}")
    else:
        plt.show()
