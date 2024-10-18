import matplotlib.pyplot as plt


def test():
    fig, ax = plt.subplots()
    im = ax.scatter(range(100), range(100))
    ax.set_xlabel("whateverrrr")
    fig.colorbar(im, ax=ax)
    plt.show()
