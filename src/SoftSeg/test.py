import matplotlib.pyplot as plt
import random


def test():
    fig, ax = plt.subplots()
    im = ax.scatter(range(100), range(100))
    ax.set_xlabel("whateverrrr")
    fig.colorbar(im, ax=ax)
    plt.show(block=False)

    fig, ax = plt.subplots()
    im = ax.scatter(range(100, 0, -1), range(100))
    ax.set_xlabel("ayooo")
    fig.colorbar(im, ax=ax)
    plt.show(block=False)

    for i in range(10**10):
        random.random()
        if i % 100 == 0:
            print(i)

    plt.show()
