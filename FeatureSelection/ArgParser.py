import argparse
import re

def check_num_files(value):
    ivalue = int(value)
    if ivalue < 0 or ivalue > 1000:
        raise argparse.ArgumentTypeError(f"{value} is out of range! Must be between 0 and 1000.")
    return ivalue

def check_num_args(features):
    print("Number of features: ", len(features))
    print(features)
    if len(features) < 1 or len(features) > 2:
        raise argparse.ArgumentTypeError("You must provide between 1 and 2 features.")
    return features

def extract_feature_names():
    with open("GetFeatureValue.py", "r") as f:
        code = f.read()
    return set(re.findall(r'case\s+"(.*?)":', code))

def create_arg_parser():
    """
    Create an argument parser with the following options:
    - good_files: Number of good files to process (0-1000)
    - faulty_gate_files: Number of faulty files to process (0-1000)
    - features: Feature(s) to process, control if the feature(s) is 
        valid according to match case in FeaturePlotting.py
    """
    valid_features = extract_feature_names()

    parser = argparse.ArgumentParser(
        description="A sample script to demonstrate argcomplete with tab completion."
    )

    parser.add_argument(
        "--good_gate_files", type=check_num_files, required=True,
        help="Number of good files to process (0-1000)"
    )
    parser.add_argument(
        "--faulty_gate_files", type=check_num_files, required=True,
        help="Number of faulty files to process (0-1000)"
    )
    parser.add_argument(
        "--features", type=str, nargs='+', choices=valid_features,
        help="Feature(s) to process, between 1 and 2 features",
    )

    return parser
