import matplotlib.pyplot as plt
import numpy as np
import os
import random
from GetFeatureValue import get_feature_value
from ArgParser import create_arg_parser

def get_files( num_of_good_gate_files, num_of_faulty_gate_files):
    good_gate_files = []
    faulty_gate_files = []

    functioning_directories = [
        "../Recording/Functioning_gate_recordings/Day 2/Session 1",
        "../Recording/Functioning_gate_recordings/Day 2/Session 2"
    ]
    faulty_directories = [

        "../Recording/Faulty_gate_recordings/Day 2/Session 1",
        "../Recording/Faulty_gate_recordings/Day 2/Session 2"

    ]

    for dir in functioning_directories:
        for entry in os.scandir(dir):
            if entry.is_file():
                good_gate_files.append(entry.path)

    for dir in faulty_directories:
        for entry in os.scandir(dir):
            if entry.is_file():
                faulty_gate_files.append(entry.path)

    while(len(good_gate_files) >  num_of_good_gate_files):
        good_gate_files.pop()

    while(len(faulty_gate_files) > num_of_faulty_gate_files):
        faulty_gate_files.pop()

    return good_gate_files, faulty_gate_files

def plot_one_dim(data1_title, data1, data1_labels, data2_title, data2, data2_labels, xlabel):
    data1_color = 'blue'
    data2_color = 'red'

    y1 = np.ones(len(data1))
    y2 = np.ones(len(data2))*-1

    fig, ax = plt.subplots()
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    sc1 = ax.scatter(data1, y1, color=data1_color, label=data1_title, picker=True)
    sc2 = ax.scatter(data2, y2, color=data2_color, label=data2_title, picker=True)

    plt.xlabel(xlabel)
    plt.ylim(-20, 20)
    plt.yticks([])
    plt.legend(loc="upper right")
    plt.grid(True)

    tooltip = ax.text(0, 0, "", fontsize=10, color="black", bbox=dict(facecolor='white', alpha=0.7))
    tooltip.set_visible(False)

    def on_hover(event):

        if event.inaxes == ax:
            cont1, ind1 = sc1.contains(event)
            cont2, ind2 = sc2.contains(event)

            if cont1:
                index = ind1["ind"][0]
                label = data1_labels[index]
            elif cont2:
                index = ind2["ind"][0]
                label = data2_labels[index]
            else:
                tooltip.set_visible(False)
                fig.canvas.draw_idle()
                return

            tooltip.set_text(label)
            tooltip.set_position((event.xdata, event.ydata))
            tooltip.set_visible(True)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    plt.show()

def plot_two_dim(data1_title, data1, data1_labels, data2_title, data2, data2_labels, xlabel, ylabel):
    data1_color = 'blue'
    data2_color = 'red'

    x1, y1 = data1[:, 0], data1[:, 1]
    x2, y2 = data2[:, 0], data2[:, 1]

    fig, ax = plt.subplots()
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    sc1 = ax.scatter(x1, y1, color=data1_color, label=data1_title, picker=True)
    sc2 = ax.scatter(x2, y2, color=data2_color, label=data2_title, picker=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc="upper right")

    tooltip = ax.text(0, 0, "", fontsize=10, color="black", bbox=dict(facecolor='white', alpha=0.7))
    tooltip.set_visible(False)

    def on_hover(event):

        if event.inaxes == ax:
            cont1, ind1 = sc1.contains(event)
            cont2, ind2 = sc2.contains(event)

            if cont1:
                index = ind1["ind"][0]
                label = data1_labels[index]
            elif cont2:
                index = ind2["ind"][0]
                label = data2_labels[index]
            else:
                tooltip.set_visible(False)
                fig.canvas.draw_idle()
                return

            tooltip.set_text(label)
            tooltip.set_position((event.xdata, event.ydata))
            tooltip.set_visible(True)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    plt.show()

def plot_one_feature(feature_type: str,  num_of_good_gate_files, num_of_faulty_gate_files):
    """Feature type is a string thats defined to specify a certain feature. If you wish to add a new type. Simply implement it in the match case in the get_feature_value function"""

    good_gate_files, faulty_gate_files = get_files( num_of_good_gate_files, num_of_faulty_gate_files)

    good_gate_feature_values = []
    faulty_gate_feature_values = []

    for file in good_gate_files:
        good_gate_feature_values.append(get_feature_value(feature_type, file))

    for file in faulty_gate_files:
        faulty_gate_feature_values.append(get_feature_value(feature_type, file))

    plot_one_dim("Func gates", good_gate_feature_values, good_gate_files, "Faulty gates", faulty_gate_feature_values, faulty_gate_files, feature_type)

def plot_two_features(feature_1_type: str, feature_2_type: str,  num_of_good_gate_files, num_of_faulty_gate_files):
    """Feature type is a string thats defined to specify a certain feature. If you wish to add a new type. Simply implement it in the match case in the get_feature_value function"""

    good_gate_files, faulty_gate_files = get_files( num_of_good_gate_files, num_of_faulty_gate_files)

    functioning_feature_1_values = []
    faulty_feature_1_values = []

    functioning_feature_2_values = []
    faulty_feature_2_values = []

    for file in good_gate_files:
        functioning_feature_1_values.append(get_feature_value(feature_1_type, file))
        functioning_feature_2_values.append(get_feature_value(feature_2_type, file))

    for file in faulty_gate_files:
        faulty_feature_1_values.append(get_feature_value(feature_1_type, file))
        faulty_feature_2_values.append(get_feature_value(feature_2_type, file))

    combinded_good_gate_feature_values = np.vstack((functioning_feature_1_values, functioning_feature_2_values)).T

    combinded_faulty_gate_feature_values = np.vstack((faulty_feature_1_values, faulty_feature_2_values)).T

    plot_two_dim("Func gates", combinded_good_gate_feature_values, good_gate_files, "Faulty gates", combinded_faulty_gate_feature_values, faulty_gate_files, feature_1_type, feature_2_type)

def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    if len(args.features) == 1:
        plot_one_feature(args.features[0], args.good_gate_files, args.faulty_gate_files)
    elif len(args.features) == 2:
        plot_two_features(args.features[0], args.features[1], args.good_gate_files, args.faulty_gate_files)
    else:
        parser.print_help()

# Exampleusage plot_two_dim
# test1 = np.random.rand(10) * 100
# test2 = np.random.rand(10) * 100
# plot_one_dim(data1=test1, data1_title="Functioning", data1_labels=[f"Point {i} test1" for i in range(len(test1))], data2=test2, data2_labels=[f"Point {i} test2" for i in range(len(test1))], data2_title="Faulty", xlabel="Feature")

# Exampleusage plot_two_dim
# test1 = np.random.rand(100, 2) * 4
# test2 = (np.random.rand(100, 2) * 4) +6
# plot_two_dim(data1=test1, data1_title="Functioning", data1_labels=[f"Point {i} test1" for i in range(len(test1))], data2=test2, data2_title="Faulty", data2_labels=[f"Point {i} test2" for i in range(len(test1))], xlabel="Feature 1", ylabel="Feature 2")

if __name__ == "__main__":
    main()