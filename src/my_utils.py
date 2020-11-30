import matplotlib.pyplot as plt


def plot_yy(x, y1, y2, ylabel1, ylabel2):

    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()
    # make a plot
    ax.plot(x, y1, color="red", marker="o")
    # set y-axis label
    ax.set_ylabel(ylabel1, color="red", fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(x, y2, color="blue", marker="o")
    ax2.set_ylabel(ylabel2, color="blue", fontsize=14)
    plt.show()

    return


def remove_mean(data):

    mean = sum(data) / len(data)
    return [(x - mean) for x in data]
