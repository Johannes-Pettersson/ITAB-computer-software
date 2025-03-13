import matplotlib.pyplot as plt
import numpy as np

def plot_one_dim(data1_title, data1, data2_title, data2, xlabel):
    data1_color = 'blue'
    data2_color = 'red'

    y1 = np.zeros(len(data1))
    y2 = np.zeros(len(data2))

    plt.scatter(data1, y1, color=data1_color, label=data1_title)
    plt.scatter(data2, y2, color=data2_color, label=data2_title)

    plt.xlabel(xlabel)
    plt.yticks([])
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.show()

def plot_two_dim(data1_title, data1, data2_title, data2, xlabel, ylabel):
    data1_color = 'blue'
    data2_color = 'red'

    x1, y1 = data1[:, 0], data1[:, 1]  # Extrahera x- och y-koordinater
    x2, y2 = data2[:, 0], data2[:, 1]  # Extrahera x- och y-koordinater

    plt.scatter(x1, y1, color=data1_color, label=data1_title)  # Rita ut punkter
    plt.scatter(x2, y2, color=data2_color, label=data2_title)  # Rita ut punkter
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()

# Exampleusage plot_two_dim
# test1 = np.random.rand(10) * 100
# test2 = np.random.rand(10) * 100
# plot_one_dim(data1=test1, data1_title="Functioning", data2=test2, data2_title="Faulty", xlabel="Feature")


# Exampleusage plot_two_dim
# test1 = np.random.rand(100, 2) * 10
# test2 = np.random.rand(100, 2) * 10 
# plot_two_dim(data1=test1, data1_title="Functioning", data2=test2, data2_title="Faulty", xlabel="Feature 1", ylabel="Feature 2")