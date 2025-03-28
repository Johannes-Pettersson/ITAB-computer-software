
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "."))) # Add the path to this folder

from SpectralCentroid import calculate_values as sc_calculate_values
from RootMeanSquareEnergy import calculate_values as rmse_calculate_values
from ZeroCrossingRate import calculate_values as zcr_calculate_values
from AmplitudeEnvelope import calculate_values as ae_calculate_values
from BandEnergyRatio import calculate_values as ber_calculate_values
from SpectralBandwidth import calculate_values as sb_calculate_values
from SpectralRolloff import calculate_values as ro_calculate_values
from MelFrequencyCepstralCoefficients import calculate_values as mfcc_calculate_values

def get_feature_value(feature_type, file):
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