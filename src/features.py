import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

def extract_features(X, fs=256):
    feature_list = []
    for segment in X:
        sig = segment[0]  # (256,)
        features = []

        # Time-domain
        features.append(np.mean(sig))
        features.append(np.std(sig))
        features.append(np.var(sig))
        features.append(np.max(sig) - np.min(sig))
        features.append(skew(sig))
        features.append(kurtosis(sig))

        # Hjorth parameters
        diff1 = np.diff(sig)
        diff2 = np.diff(diff1)
        var_zero = np.var(sig)
        var_d1 = np.var(diff1) if len(diff1) > 0 else 0
        var_d2 = np.var(diff2) if len(diff2) > 0 else 0
        features.append(var_zero)  # Activity
        features.append(np.sqrt(var_d1/var_zero) if var_zero > 0 else 0)  # Mobility
        features.append(np.sqrt(var_d2/var_d1) / np.sqrt(var_d1/var_zero) if var_d1 > 0 and var_zero > 0 else 0)  # Complexity

        # Frequency-domain (band power)
        freqs, psd = welch(sig, fs=fs)
        bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
        for (low, high) in bands.values():
            band_power = np.sum(psd[(freqs >= low) & (freqs <= high)])
            features.append(band_power)

        feature_list.append(features)

    return np.array(feature_list)
