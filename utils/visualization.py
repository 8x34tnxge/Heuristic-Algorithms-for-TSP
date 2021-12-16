import matplotlib.pyplot as plt


def visualize(schedule, dataLoader, valueWatcher, fileName, save=False):
    validValue = valueWatcher[1:]
    bestValue= valueWatcher[-1]
    fig = plt.figure()

    plt.subplot(2, 1, 1)
    plt.title(f"best value: {bestValue:.6f}")
    plt.plot(validValue)

    plt.subplot(2, 1, 2)
    plt.title('路线图')
    plt.xlabel('')
    plt.ylabel('')
    x = []
    y = []
    for idInfo, geoInfo in dataLoader:
        x.append(geoInfo.x)
        y.append(geoInfo.y)
    plt.scatter(x, y)
    schedule.append(schedule[0])
    for prev in range(len(schedule)-1):
        next = prev + 1
        plt.plot([dataLoader[schedule[prev]].x, dataLoader[schedule[next]].x], [dataLoader[schedule[prev]].y, dataLoader[schedule[next]].y])

    if save:
        fig.savefig(f"./result/{fileName}")
    else:
        plt.show()
