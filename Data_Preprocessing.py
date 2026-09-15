#Data Preprocessing
import pandas as pd
import numpy as np

combained_file = "combined_marketing_campaign_data.csv"
cleaned_file = "cleaned_marketing_campaign_data.csv"

campaign_df = pd.read_csv(combained_file)

# Remove duplicates
if "Campaign_ID" in campaign_df.columns:
    campaign_df.drop_duplicates(
        subset="Campaign_ID",
        keep="first",
        inplace=True
    )
else:
    campaign_df.drop_duplicates(
        inplace=True
    )

campaign_df.reset_index(
    drop=True,
    inplace=True
)

# Convert Date
if "Date" in campaign_df.columns:
    campaign_df["Date"] = pd.to_datetime(
        campaign_df["Date"],
        dayfirst=True,
        errors="coerce"
    )

# Clean numerical columns
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

        campaign_df[col] = campaign_df[col].fillna(
            median
        )

# Clean categorical columns
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
            campaign_df[col] = campaign_df[col].fillna(
                mode.iloc[0]
            )
        else:
            campaign_df[col] = campaign_df[col].fillna(
                "Unknown"
            )

        campaign_df[col] = campaign_df[col].str.title()

# Clean Campaign ID
if "Campaign_ID" in campaign_df.columns:

    campaign_df["Campaign_ID"] = (
        campaign_df["Campaign_ID"]
        .astype("string")
        .str.strip()
    )

    campaign_df["Campaign_ID"] = (
        campaign_df["Campaign_ID"]
        .fillna("")
    )

    missing_id = campaign_df["Campaign_ID"] == ""

    campaign_df.loc[
        missing_id,
        "Campaign_ID"
    ] = [
        f"AUTO_{i}"
        for i in range(
            1,
            missing_id.sum() + 1
        )
    ]

# Remove negative values
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
        campaign_df[col] = campaign_df[col].clip(
            lower=0
        )

# Apply business rules
required_columns = [
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions"
]

if all(
    col in campaign_df.columns
    for col in required_columns
):

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

# Validate ROI

if "ROI" in campaign_df.columns:

    # Check for invalid infinite values
    campaign_df["ROI"] = campaign_df["ROI"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing ROI with median
    median_roi = campaign_df["ROI"].median()

    if pd.isna(median_roi):
        median_roi = 0

    campaign_df["ROI"] = campaign_df["ROI"].fillna(
        median_roi
    )

    # Round ROI
    campaign_df["ROI"] = campaign_df["ROI"].round(2)

    # Display ROI range for validation
    print("\nROI Validation:")
    print("Minimum ROI:", campaign_df["ROI"].min())
    print("Maximum ROI:", campaign_df["ROI"].max())
    print("Negative ROI Count:", (campaign_df["ROI"] < 0).sum())

# Convert integer columns
integer_columns = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions"
]

for col in integer_columns:

    if col in campaign_df.columns:

        campaign_df[col] = campaign_df[col].round().astype(int)

# Convert float columns
float_columns = [
    "Revenue",
    "Acquisition_Cost",
    "ROI",
    "Engagement_Score"
]

for col in float_columns:

    if col in campaign_df.columns:
        campaign_df[col] = campaign_df[col].astype(float)

# Final validation
print("\nFinal Dataset Shape:")
print(campaign_df.shape)

print("\nMissing Values:")
print(campaign_df.isnull().sum())

print("\nDuplicate Records:")
print(campaign_df.duplicated().sum())

print("\nData Types:")
print(campaign_df.dtypes)

# Save cleaned dataset
campaign_df.to_csv(
    cleaned_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nCleaned Dataset Saved Successfully")
