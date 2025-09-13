import pandas as pd
import numpy as np

def load_eeg_file_for_gui(filepath):
    """
    Load file and return (numeric_dataframe, channel_names).
    Keeps only numeric columns (drops timestamps/strings).
    """
    if filepath.lower().endswith(".csv"):
        df = pd.read_csv(filepath, header=0, low_memory=False)
    else:
        df = pd.read_excel(filepath, header=0)

    # If first column name or values suggest timestamp, drop if non-numeric
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # If there's no numeric columns, try to coerce everything then fill NaN with 0
    if numeric_df.shape[1] == 0:
        coerced = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        numeric_df = coerced

    # Reset column names to simple ones if necessary
    numeric_df.columns = [str(c) for c in numeric_df.columns]

    return numeric_df, list(numeric_df.columns)
