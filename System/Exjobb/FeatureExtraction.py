import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Add the path to the root folder of the git repository
from FeatureSelection.GetFeatureValue import get_feature_value # type: ignore


class FeatureExtraction:
    def __init__(self, feature_list, files):
        self.feature_list = feature_list
        self.features = {feature: [] for feature in feature_list}
        self.extract_features_from_files(files)

    def extract_features_from_file(self, file):
        for feature in self.feature_list:
            value = get_feature_value(feature, file)
            self.features[feature].append(value)

    def extract_features_from_files(self, file_list):
        for file in file_list:
            self.extract_features_from_file(file)

    def __getitem__(self, feature_name):
        return self.features.get(feature_name, [])
