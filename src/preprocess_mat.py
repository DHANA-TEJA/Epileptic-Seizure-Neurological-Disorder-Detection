import os
import numpy as np
from scipy.io import loadmat

def segment_signal(signal, window_size=256, step_size=128):
    """Split EEG signal into overlapping windows"""
    segments = []
    n_samples = signal.shape[0]

    for start in range(0, n_samples - window_size, step_size):
        end = start + window_size
        window = signal[start:end, :].T   # (channels=1, window_size)
        segments.append(window)

    return np.array(segments)

def load_mat_dataset(base_path="data/raw/mat_data"):
    """
    Loads EEG data from ictal, interictal, preictal folders.
    Returns: X (segments), y (labels)
    """
    label_map = {"ictal": 0, "interictal": 1, "preictal": 2}
    X, y = [], []

    for label_name, label_id in label_map.items():
        folder = os.path.join(base_path, label_name)
        if not os.path.exists(folder):
            print(f"⚠️ Skipping {label_name} (folder not found)")
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".mat")]
        for file in files:
            mat_data = loadmat(os.path.join(folder, file))

            # Each .mat file should have a key same as the folder name
            if label_name not in mat_data:
                print(f"⚠️ Key '{label_name}' not found in {file}. Keys: {mat_data.keys()}")
                continue

            eeg_signal = mat_data[label_name]   # (samples, 1)
            segments = segment_signal(eeg_signal)
            X.append(segments)
            y.extend([label_id] * len(segments))

    if len(X) == 0:
        raise ValueError("❌ No data loaded. Check folder names and .mat file keys.")

    X = np.vstack(X)   # shape: (n_segments, channels, window_size)
    y = np.array(y)
    return X, y
