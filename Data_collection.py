# Data Collection & Understanding
import os
import pandas as pd

DATASETS = {
    "Nykaa": "nykaa_campaign_data_with_nulls.csv",
    "Purplle": "purplle_campaign_data_with_nulls.csv",
    "Tira": "tira_campaign_data_with_nulls.csv"
}

combined_file = "combined_marketing_campaign_data.csv"

dataframes = []

for brand, file_name in DATASETS.items():

    if not os.path.exists(file_name):
        raise FileNotFoundError(
            f"Dataset not found: {file_name}"
        )

    print(f"\nLoading {brand} Dataset...")

    df = pd.read_csv(file_name)

    df["Brand"] = brand

    print(f"{brand} Shape: {df.shape}")

    dataframes.append(df)

print("\nAll Datasets Loaded Successfully.")

campaign_df = pd.concat(
    dataframes,
    ignore_index=True
)

print("\nDatasets Merged Successfully.")

print("\nDataset Information:")
print(campaign_df.info())

print("\nDataset Shape:")
print(campaign_df.shape)

print("\nColumn Names:")
print(campaign_df.columns.tolist())

print("\nData Types:")
print(campaign_df.dtypes)

print("\nMissing Values:")
print(campaign_df.isnull().sum())

print("\nduplicate_count:")
print(campaign_df.duplicated().sum())

print("\nFirst Five Records:")
print(campaign_df.head())

print("\nStatistical Summary:")
print(campaign_df.describe(include="all"))

print("\nCampaign Count by Brand:")
print(campaign_df["Brand"].value_counts())

campaign_df.to_csv(
    combined_file,
    index=False,
    encoding="utf-8-sig"
)
print("\nCombined Dataset Saved Successfully.")
