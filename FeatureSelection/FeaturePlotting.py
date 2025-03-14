import matplotlib.pyplot as plt
import numpy as np
import os
import random
from SpectralCentroid import calculate_values as sc_calculate_values

def get_files(num_of_functioning_files, num_of_faulty_files):
    functioning_files = []
    faulty_files = []

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
                functioning_files.append(entry.path)

    for dir in faulty_directories:
        for entry in os.scandir(dir):  
            if entry.is_file():
                faulty_files.append(entry.path)    

    while(len(functioning_files) > num_of_functioning_files):
        functioning_files.pop(random.randrange(len(functioning_files)))

    while(len(faulty_files) > num_of_faulty_files):
        faulty_files.pop(random.randrange(len(faulty_files)))

    return functioning_files, faulty_files

def plot_one_dim(data1_title, data1, data1_labels, data2_title, data2, data2_labels, xlabel):
    data1_color = 'blue'
    data2_color = 'red'

    y1 = np.ones(len(data1))
    y2 = np.ones(len(data2))*-1

    fig, ax = plt.subplots()

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

def _get_feature_value(feature_type, file):
    match feature_type:
        case "sc_min":
            _, _, _, min_val, _, _, _, _, _ = sc_calculate_values(file)
            return min_val
        case "sc_max":
            _, _, _, _, max_val, _, _, _, _ = sc_calculate_values(file)
            return max_val
        case "sc_ptp":
            _, _, _, _, _, ptp_val, _, _, _ = sc_calculate_values(file)
            return ptp_val
        case "sc_deriv_max":
            _, _, _, _, _, _, _, max_deriv, _ = sc_calculate_values(file)
            return max_deriv
        case "sc_deriv_min":
            _, _, _, _, _, _, _, _, min_deriv = sc_calculate_values(file)
            return min_deriv
        case default:
            raise Exception("Feature_type not defined")
        

def plot_one_feature(feature_type: str, num_of_functioning_files, num_of_faulty_files):
    """Feature type is a string thats defined to specify a certain feature. If you wish to add a new type. Simply implement it in the match case in the _get_feature_value function"""       
    
    functioning_files, faulty_files = get_files(num_of_functioning_files, num_of_faulty_files)

    functioning_feature_values = []
    faulty_feature_values = []

    for file in functioning_files:
        functioning_feature_values.append(_get_feature_value(feature_type, file))

    for file in faulty_files:
        faulty_feature_values.append(_get_feature_value(feature_type, file))
    
    plot_one_dim("Func gates", functioning_feature_values, functioning_files, "Faulty gates", faulty_feature_values, faulty_files, feature_type)
    
def plot_two_features(feature_1_type: str, feature_2_type: str, num_of_functioning_files, num_of_faulty_files):
    """Feature type is a string thats defined to specify a certain feature. If you wish to add a new type. Simply implement it in the match case in the _get_feature_value function"""       

    functioning_files, faulty_files = get_files(num_of_functioning_files, num_of_faulty_files)

    functioning_feature_1_values = []
    faulty_feature_1_values = []

    functioning_feature_2_values = []
    faulty_feature_2_values = []

    for file in functioning_files:
        functioning_feature_1_values.append(_get_feature_value(feature_1_type, file))
        functioning_feature_2_values.append(_get_feature_value(feature_2_type, file))

    for file in faulty_files:
        faulty_feature_1_values.append(_get_feature_value(feature_1_type, file))
        faulty_feature_2_values.append(_get_feature_value(feature_2_type, file))

    combinded_functioning_feature_values = np.vstack((functioning_feature_1_values, functioning_feature_2_values)).T

    combinded_faulty_feature_values = np.vstack((faulty_feature_1_values, faulty_feature_2_values)).T

    plot_two_dim("Func gates", combinded_functioning_feature_values, functioning_files, "Faulty gates", combinded_faulty_feature_values, faulty_files, feature_1_type, feature_2_type)



# Exampleusage plot_two_dim
# test1 = np.random.rand(10) * 100
# test2 = np.random.rand(10) * 100
# plot_one_dim(data1=test1, data1_title="Functioning", data1_labels=[f"Point {i} test1" for i in range(len(test1))], data2=test2, data2_labels=[f"Point {i} test2" for i in range(len(test1))], data2_title="Faulty", xlabel="Feature")


# Exampleusage plot_two_dim
# test1 = np.random.rand(100, 2) * 4
# test2 = (np.random.rand(100, 2) * 4) +6  
# plot_two_dim(data1=test1, data1_title="Functioning", data1_labels=[f"Point {i} test1" for i in range(len(test1))], data2=test2, data2_title="Faulty", data2_labels=[f"Point {i} test2" for i in range(len(test1))], xlabel="Feature 1", ylabel="Feature 2")


# plot_one_feature("sc_ptp", 10000, 10000)
plot_two_features(feature_1_type="sc_ptp", feature_2_type="sc_deriv_min", num_of_functioning_files=10000, num_of_faulty_files=10000)