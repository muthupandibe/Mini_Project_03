# ==============================================================
# DATA COLLECTION
# ==============================================================
import os
import pandas as pd

# ==============================================================
# FILE PATHS
# ==============================================================

DATASETS = {
    "Nykaa": "nykaa_campaign_data_with_nulls.csv",
    "Purplle": "purplle_campaign_data_with_nulls.csv",
    "Tira": "tira_campaign_data_with_nulls.csv"
}

OUTPUT_FILE = "combined_marketing_campaign_data.csv"

# ==============================================================
# LOAD DATASETS
# ==============================================================

dataframes = []

print("=" * 70)
print("STEP 1 : DATA COLLECTION")
print("=" * 70)

for brand, file_name in DATASETS.items():

    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Dataset not found : {file_name}")

    print(f"\nLoading {brand} Dataset...")

    df = pd.read_csv(file_name)

    # Add Brand Column
    df["Brand"] = brand

    print(f"{brand} Shape : {df.shape}")

    dataframes.append(df)

print("\nAll Datasets Loaded Successfully.")

# ==============================================================
# MERGE DATASETS
# ==============================================================

campaign_df = pd.concat(
    dataframes,
    ignore_index=True
)

print("\nDatasets Merged Successfully")

# ==============================================================
# BASIC INFORMATION
# ==============================================================

print("\nDataset Shape")
print(campaign_df.shape)

print("\nRows :", campaign_df.shape[0])
print("Columns :", campaign_df.shape[1])

# ==============================================================
# COLUMN NAMES
# ==============================================================

print("\nColumn Names")

for col in campaign_df.columns:
    print(col)

# ==============================================================
# DATA TYPES
# ==============================================================

print("\nData Types")

print(campaign_df.dtypes)

# ==============================================================
# MISSING VALUES
# ==============================================================

print("\nMissing Values")

print(campaign_df.isnull().sum())

# ==============================================================
# DUPLICATES
# ==============================================================

duplicate_count = campaign_df.duplicated().sum()

print("\nDuplicate Records :", duplicate_count)

# ==============================================================
# SAMPLE DATA
# ==============================================================

print("\nFirst Five Records")

print(campaign_df.head())

print("\nLast Five Records")

print(campaign_df.tail())

# ==============================================================
# STATISTICAL SUMMARY
# ==============================================================

print("\nStatistical Summary")

print(campaign_df.describe(include="all"))

# ==============================================================
# BRAND SUMMARY
# ==============================================================

print("\nCampaign Count by Brand")

print(campaign_df["Brand"].value_counts())

# ==============================================================
# SAVE DATASET
# ==============================================================

campaign_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nCombined Dataset Saved Successfully")

print(f"Output File : {OUTPUT_FILE}")

