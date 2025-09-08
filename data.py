# import os
# import numpy as np
# from scipy.io import loadmat

# # Path to your dataset root
# root_dir = r"Data\EEG Epilepsy Datasets"

# # Map folder name -> label
# label_map = {
#     "ictal": 1,
#     "interictal": 0,
#     "preictal": 2
# }

# X = []   # features
# y = []   # labels
# shapes = set()  # to check homogeneity

# # Loop over each folder
# for folder, label in label_map.items():
#     folder_path = os.path.join(root_dir, folder)
#     if not os.path.exists(folder_path):
#         print(f"⚠️ Skipping missing folder: {folder_path}")
#         continue
    
#     for f in os.listdir(folder_path):
#         if f.endswith(".mat"):
#             file_path = os.path.join(folder_path, f)
#             data = loadmat(file_path)

#             # Find the first variable that isn't __metadata__
#             for key in data:
#                 if not key.startswith("__"):
#                     arr = data[key]
                    
#                     # Ensure it's 2D (e.g. (1024,1) -> flatten)
#                     arr = np.array(arr).squeeze()
                    
#                     shapes.add(arr.shape)
                    
#                     X.append(arr)
#                     y.append(label)
#                     break  # assume only one useful variable per file

# # Convert to numpy arrays
# X = np.array(X, dtype=object)  # object dtype in case lengths differ
# y = np.array(y)

# # ---------------------------
# # Check results
# # ---------------------------
# print("\nUnique shapes found:", shapes)
# print("Total samples:", len(X))
# print("Labels distribution:", {val: list(y).count(val) for val in set(y)})

# # Optional: if all shapes are the same, you can stack into 2D array
# if len(shapes) == 1:
#     X = np.stack(X, axis=0)
#     print("✅ Data stacked into shape:", X.shape)
# else:
#     print("⚠️ Different shapes found, need preprocessing (resampling/truncating).")

# np.save("X.npy", X)
# np.save("y.npy", y)
# print("✅ Saved X.npy and y.npy")

# import pandas as pd

# # ===============================
# # 1. Load EEG dataset
# # ===============================
# print("📂 Loading EEG data...")
# combined = pd.read_csv(r"Data\SNMC_dataport\combined_eeg.csv")

# print("✅ EEG data loaded:", combined.shape)
# print("Columns:", combined.columns.tolist())


# # ===============================
# # 2. Load Annotations
# # ===============================
# print("\n📂 Loading annotations...")
# annotations = pd.read_csv(r"Data\SNMC_dataport\Annotations_SNMC.csv")

# # Normalize column names
# annotations.columns = annotations.columns.str.strip().str.lower()

# # Keep only patient_id + seizure info
# annotations = annotations.rename(
#     columns={
#         "patient number": "patient_id",
#         "seizure recorded on eeg": "seizure_label"
#     }
# )[["patient_id", "seizure_label"]]

# print("✅ Cleaned annotations:\n", annotations.head())


# # ===============================
# # 3. Clean patient_id in EEG data
# # ===============================
# # Convert "Patient10" → 10
# combined["patient_id"] = combined["patient_id"].str.replace("Patient", "", regex=False).astype(int)


# # ===============================
# # 4. Merge EEG + Annotations
# # ===============================
# final = combined.merge(annotations, on="patient_id", how="left")

# print("\n✅ Final dataset shape:", final.shape)
# print("Columns:", final.columns.tolist())
# print("🔎 Seizure label distribution:\n", final["seizure_label"].value_counts(dropna=False))


# # ===============================
# # 5. Save Final Dataset
# # ===============================
# final_csv = r"Data\SNMC_dataport\final.csv"
# final_parquet = r"Data\SNMC_dataport\final.parquet"

# final.to_csv(final_csv, index=False)
# final.to_parquet(final_parquet, index=False)

# print(f"\n💾 Saved final dataset to:\n- {final_csv}\n- {final_parquet}")
# final["seizure_binary"] = final["seizure_label"].map({"Yes": 1, "No": 0})


# import pandas as pd

# # Load your already saved final dataset
# final = pd.read_csv("Data/SNMC_dataport/final.csv")

# # Add binary label column
# final["seizure_binary"] = final["seizure_label"].map({"Yes": 1, "No": 0})

# # Save again (overwrite or new file)
# final.to_csv("Data/SNMC_dataport/final.csv", index=False)

# print("✅ Added seizure_binary column and re-saved.")
# print(final.head())

import pandas as pd

# Load the final saved CSV
final = pd.read_csv("Data/SNMC_dataport/final.csv", low_memory=False)

# 1. Remove the junk row where Time == "(HH-MM-SS)"
final = final[final["Time"] != "(HH-MM-SS)"]

# 2. Drop rows with missing seizure labels (keep only labeled data for training)
final = final.dropna(subset=["seizure_binary"])

# 3. Reset index after dropping rows
final = final.reset_index(drop=True)

# 4. Save cleaned dataset
final.to_csv("Data/SNMC_dataport/final_clean.csv", index=False)

print("✅ Cleaned dataset saved as final_clean.csv and final_clean.parquet")
print(final.head())
print(f"Shape after cleaning: {final.shape}")
