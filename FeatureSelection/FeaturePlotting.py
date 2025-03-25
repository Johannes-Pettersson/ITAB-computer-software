import matplotlib.pyplot as plt
import numpy as np
import os
import random
from SpectralCentroid import calculate_values as sc_calculate_values
from RootMeanSquareEnergy import calculate_values as rmse_calculate_values
from ZeroCrossingRate import calculate_values as zcr_calculate_values
from AmplitudeEnvelope import calculate_values as ae_calculate_values
from BandEnergyRatio import calculate_values as ber_calculate_values
from SpectralBandwidth import calculate_values as sb_calculate_values
from SpectralRolloff import calculate_values as ro_calculate_values
from MelFrequencyCepstralCoefficients import calculate_values as mfcc_calculate_values
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
    fig.set_size_inches(15,8)

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
    fig.set_size_inches(15, 8)

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

    plt.savefig(fr"C:\Users\johan\OneDrive - Jonkoping University\DIS3\Examensarbete\Feature prestudy images\evaluation\{xlabel}\{xlabel}_{ylabel}")
    plt.close()

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
        case "rmse_mean":
            _, _, _, _, mean_val, _, _ = rmse_calculate_values(file)
            return mean_val
        case "rmse_max":
            _, _, _, _, _, max_val, _ = rmse_calculate_values(file)
            return max_val
        case "rmse_std":
            _, _, _, _, _, _, std_val = rmse_calculate_values(file)
            return std_val
        case "zcr_total":
            _, _, _, total_val, _, _, _, _ = zcr_calculate_values(file)
            return total_val
        case "zcr_mean":
            _, _, _, _, _, mean_val, _, _ = zcr_calculate_values(file)
            return mean_val
        case "zcr_max":
            _, _, _, _, _, _, max_val, _ = zcr_calculate_values(file)
            return max_val
        case "zcr_std":
            _, _, _, _, _, _, _, std_val = zcr_calculate_values(file)
            return std_val
        case "ae_mean":
            _, _, _, _, mean_val, _, _ = ae_calculate_values(file)
            return mean_val
        case "ae_max":
            _, _, _, _, _, max_val, _ = ae_calculate_values(file)
            return max_val
        case "ae_std":
            _, _, _, _, _, _, std_val = ae_calculate_values(file)
            return std_val
        case "sb_max":
            _, _, _, _, _, sb_max, _, _ = sb_calculate_values(file)
            return sb_max
        case "sb_min":
            _, _, _, sb_min, _, _, _, _ = sb_calculate_values(file)
            return sb_min
        case "sb_ptp":
            _, _, _, _, sb_ptp, _, _, _ = sb_calculate_values(file)
            return sb_ptp
        case "sb_mean":
             _, _, _, _, _, _, sb_mean, _ = sb_calculate_values(file)
             return sb_mean
        case "sb_std":
            _, _, _, _, _, _, _, sb_std = sb_calculate_values(file)
            return sb_std
        case "ro_max":
            _, _, _, _, ro_max, _, _, _ = ro_calculate_values(file=file, roll_percent=.37)
            return ro_max
        case "ro_min":
            _, _, _, _, _, ro_min, _, _ = ro_calculate_values(file=file, roll_percent=.37)
            return ro_min
        case "ro_mean":
            _, _, _, _, _, _, ro_mean, _ = ro_calculate_values(file=file, roll_percent=.37)
            return ro_mean    
        case "ro_std":
            _, _, _, _, _, _, _, ro_std = ro_calculate_values(file=file, roll_percent=.37)
            return ro_std
        case "ber_max":
            _, _, _, _, ber_max, _, _, _ = ber_calculate_values(file, 1000)
            return ber_max
        case "ber_min":
            _, _, _, _, _, ber_min, _, _ = ber_calculate_values(file, 1000)
            return ber_min
        case "ber_mean":
            _, _, _, _, _, _, ber_mean, _ = ber_calculate_values(file, 1000)
            return ber_mean
        case "ber_std":
            _, _, _, _, _, _, _, ber_std = ber_calculate_values(file, 1000)
            return ber_std
        case "mfcc_skewness":
            _, _, _, mfcc_skewness, _ = mfcc_calculate_values(file, coef=1, dct_type=4)
            return mfcc_skewness
        case "mfcc_kurtosis":
            _, _, _, _, mfcc_kurtosis = mfcc_calculate_values(file, coef=1, dct_type=4)
            return mfcc_kurtosis
        case default:
            raise Exception("Feature_type not defined")

def plot_one_feature(feature_type: str,  num_of_good_gate_files, num_of_faulty_gate_files):
    """Feature type is a string thats defined to specify a certain feature. If you wish to add a new type. Simply implement it in the match case in the _get_feature_value function"""       
    
    good_gate_files, faulty_gate_files = get_files( num_of_good_gate_files, num_of_faulty_gate_files)

    good_gate_feature_values = []
    faulty_gate_feature_values = []

    for file in good_gate_files:
        good_gate_feature_values.append(_get_feature_value(feature_type, file))

    for file in faulty_gate_files:
        faulty_gate_feature_values.append(_get_feature_value(feature_type, file))
    
    plot_one_dim("Func gates", good_gate_feature_values, good_gate_files, "Faulty gates", faulty_gate_feature_values, faulty_gate_files, feature_type)
    
def plot_two_features(feature_1_type: str, feature_2_type: str,  num_of_good_gate_files, num_of_faulty_gate_files):
    """Feature type is a string thats defined to specify a certain feature. If you wish to add a new type. Simply implement it in the match case in the _get_feature_value function"""       

    good_gate_files, faulty_gate_files = get_files( num_of_good_gate_files, num_of_faulty_gate_files)

    functioning_feature_1_values = []
    faulty_feature_1_values = []

    functioning_feature_2_values = []
    faulty_feature_2_values = []

    for file in good_gate_files:
        functioning_feature_1_values.append(_get_feature_value(feature_1_type, file))
        functioning_feature_2_values.append(_get_feature_value(feature_2_type, file))

    for file in faulty_gate_files:
        faulty_feature_1_values.append(_get_feature_value(feature_1_type, file))
        faulty_feature_2_values.append(_get_feature_value(feature_2_type, file))

    combinded_good_gate_feature_values = np.vstack((functioning_feature_1_values, functioning_feature_2_values)).T

    combinded_faulty_gate_feature_values = np.vstack((faulty_feature_1_values, faulty_feature_2_values)).T

    plot_two_dim("Func gates", combinded_good_gate_feature_values, good_gate_files, "Faulty gates", combinded_faulty_gate_feature_values, faulty_gate_files, feature_1_type, feature_2_type)

def main():
    feat1 = "sb_mean"
    feat_list = ["ae_max", "sc_ptp", "mfcc_skewness", "sb_max", "ber_min", "sc_deriv_min", "sc_max", "zcr_std", "ro_mean", "ber_max", "ber_std", "ro_max", "zcr_max", "sc_deriv_max", "ro_min"]

    while len(feat_list) > 0:
        for feat in feat_list:
            plot_two_features(feature_1_type=feat1, feature_2_type=feat, num_of_good_gate_files=700, num_of_faulty_gate_files=700)
            print(f"done with {feat1}_{feat}")

        feat1 = feat_list.pop(0)
    
    # parser = create_arg_parser()
    # args = parser.parse_args()

    # if len(args.features) == 1:
    #     plot_one_feature(args.features[0], args.good_gate_files, args.faulty_gate_files)
    # elif len(args.features) == 2:
    #     plot_two_features(args.features[0], args.features[1], args.good_gate_files, args.faulty_gate_files)
    # else:
    #     parser.print_help()

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