# ==========================================================
# DATA PREPROCESSING
# ==========================================================

import pandas as pd
import numpy as np

# ==========================================================
# LOAD DATASET
# ==========================================================

INPUT_FILE = "combined_marketing_campaign_data.csv"
OUTPUT_FILE = "cleaned_marketing_campaign_data.csv"

campaign_df = pd.read_csv(INPUT_FILE)

print("\nDataset Loaded Successfully")
print("Shape :", campaign_df.shape)

# ==========================================================
# DATASET INFORMATION
# ==========================================================

print("\nMissing Values Before Cleaning")
print(campaign_df.isnull().sum())

print("\nDuplicate Records Before Cleaning")
print(campaign_df.duplicated().sum())

# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

if "Campaign_ID" in campaign_df.columns:

    campaign_df.drop_duplicates(
        subset="Campaign_ID",
        keep="first",
        inplace=True
    )

else:

    campaign_df.drop_duplicates(inplace=True)

campaign_df.reset_index(drop=True, inplace=True)

# ==========================================================
# CONVERT DATE
# ==========================================================

if "Date" in campaign_df.columns:

    campaign_df["Date"] = pd.to_datetime(
        campaign_df["Date"],
        errors="coerce"
    )

# ==========================================================
# NUMERICAL COLUMNS
# ==========================================================

numerical_columns = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "ROI",
    "Engagement_Score"
]

for col in numerical_columns:

    if col in campaign_df.columns:

        campaign_df[col] = pd.to_numeric(
            campaign_df[col],
            errors="coerce"
        )

        campaign_df[col] = campaign_df[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

        median = campaign_df[col].median()

        if pd.isna(median):
            median = 0

        campaign_df[col] = campaign_df[col].fillna(median)

# ==========================================================
# CATEGORICAL COLUMNS
# ==========================================================

categorical_columns = [
    "Campaign_Type",
    "Target_Audience",
    "Channel_Used",
    "Language",
    "Customer_Segment",
    "Brand"
]

for col in categorical_columns:

    if col in campaign_df.columns:

        campaign_df[col] = (
            campaign_df[col]
            .astype("string")
            .str.strip()
        )

        campaign_df[col] = campaign_df[col].replace(
            ["", "nan", "None"],
            pd.NA
        )

        mode = campaign_df[col].mode()

        if not mode.empty:
            campaign_df[col] = campaign_df[col].fillna(mode.iloc[0])
        else:
            campaign_df[col] = campaign_df[col].fillna("Unknown")

        campaign_df[col] = campaign_df[col].str.title()

# ==========================================================
# CLEAN CAMPAIGN ID
# ==========================================================

if "Campaign_ID" in campaign_df.columns:

    campaign_df["Campaign_ID"] = (
        campaign_df["Campaign_ID"]
        .astype("string")
        .str.strip()
    )

    campaign_df["Campaign_ID"] = campaign_df["Campaign_ID"].fillna("")

    mask = campaign_df["Campaign_ID"] == ""

    campaign_df.loc[
        mask,
        "Campaign_ID"
    ] = [
        f"AUTO_{i}"
        for i in range(1, mask.sum() + 1)
    ]

# ==========================================================
# REMOVE NEGATIVE VALUES
# ==========================================================

positive_columns = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "Engagement_Score"
]

for col in positive_columns:

    if col in campaign_df.columns:

        campaign_df[col] = campaign_df[col].clip(lower=0)

# ==========================================================
# BUSINESS RULE VALIDATION
# ==========================================================

required = [
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions"
]

if all(col in campaign_df.columns for col in required):

    campaign_df["Clicks"] = np.minimum(
        campaign_df["Clicks"],
        campaign_df["Impressions"]
    )

    campaign_df["Leads"] = np.minimum(
        campaign_df["Leads"],
        campaign_df["Clicks"]
    )

    campaign_df["Conversions"] = np.minimum(
        campaign_df["Conversions"],
        campaign_df["Leads"]
    )

# ==========================================================
# ROI VALIDATION
# ==========================================================

if "ROI" in campaign_df.columns:

    median_roi = campaign_df["ROI"].median()

    campaign_df.loc[
        (campaign_df["ROI"] < -100) |
        (campaign_df["ROI"] > 1000),
        "ROI"
    ] = median_roi

    campaign_df["ROI"] = campaign_df["ROI"].round(2)

# ==========================================================
# SAFE INTEGER CONVERSION
# ==========================================================

integer_columns = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions"
]

for col in integer_columns:

    if col in campaign_df.columns:

        campaign_df[col] = pd.to_numeric(
            campaign_df[col],
            errors="coerce"
        )

        campaign_df[col] = campaign_df[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

        median = campaign_df[col].median()

        if pd.isna(median):
            median = 0

        campaign_df[col] = campaign_df[col].fillna(median)

        campaign_df[col] = campaign_df[col].round()

        campaign_df[col] = campaign_df[col].astype(int)

# ==========================================================
# FLOAT CONVERSION
# ==========================================================

float_columns = [
    "Revenue",
    "Acquisition_Cost",
    "ROI",
    "Engagement_Score"
]

for col in float_columns:

    if col in campaign_df.columns:

        campaign_df[col] = campaign_df[col].astype(float)

# ==========================================================
# FINAL VALIDATION
# ==========================================================

print("\nDataset Shape :", campaign_df.shape)

print("\nMissing Values")
print(campaign_df.isnull().sum())

print("\nDuplicate Records")
print(campaign_df.duplicated().sum())

print("\nData Types")
print(campaign_df.dtypes)

print("\nStatistical Summary")
print(campaign_df.describe(include="all"))

# ==========================================================
# SAVE CLEANED DATASET
# ==========================================================

campaign_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nCleaned Dataset Saved Successfully")
print("Output File :", OUTPUT_FILE)
