from LOF import calc_lof, calc_and_plot_lof
from GetFiles import get_files
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Add the path to the root folder of the git repository
from FeatureSelection.GetFeatureValue import get_feature_value
import numpy as np

def get_feature_value_list(good_gate_files, faulty_gate_files, feature_1_type, feature_2_type):
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

    return combinded_good_gate_feature_values, combinded_faulty_gate_feature_values

def main():
    good_gate_files, faulty_gate_files = get_files(1,100)


    good_gate_feature_values, faulty_gate_feature_values = get_feature_value_list(good_gate_files=good_gate_files, faulty_gate_files=faulty_gate_files, feature_1_type="sc_min", feature_2_type="sb_max")
    calc_and_plot_lof(faulty_gate_feature_values[:-1], faulty_gate_feature_values[-1])

if __name__ == "__main__":
    main()