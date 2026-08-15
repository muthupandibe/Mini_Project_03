#Feature Engineering
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

cleaned_file = "cleaned_marketing_campaign_data.csv"
featured_file = "feature_engineered_marketing_campaign_data.csv"

campaign_df = pd.read_csv(cleaned_file)

def safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(0)

print("\nCreating Engineered Features...")

campaign_df["CTR"] = safe_divide(
    campaign_df["Clicks"],
    campaign_df["Impressions"]
)

campaign_df["Conversion_Rate"] = safe_divide(
    campaign_df["Conversions"],
    campaign_df["Clicks"]
)

campaign_df["Cost_Per_Click"] = safe_divide(
    campaign_df["Acquisition_Cost"],
    campaign_df["Clicks"]
)

campaign_df["Cost_Per_Conversion"] = safe_divide(
    campaign_df["Acquisition_Cost"],
    campaign_df["Conversions"]
)

campaign_df["Lead_Conversion_Rate"] = safe_divide(
    campaign_df["Conversions"],
    campaign_df["Leads"]
)

engineered_features = [
    "CTR",
    "Conversion_Rate",
    "Cost_Per_Click",
    "Cost_Per_Conversion",
    "Lead_Conversion_Rate"
]

campaign_df[engineered_features] = campaign_df[
    engineered_features
].round(4)

print("Engineered Features Created Successfully")

print("\nCreating Profit_Flag...")

campaign_df["Profit_Flag"] = np.where(
    campaign_df["ROI"] > 0,
    1,
    0
).astype(int)

print("Profit_Flag Created Successfully")

print("\nProfit_Flag Distribution:")
print(campaign_df["Profit_Flag"].value_counts())

print("\nEncoding Channel_Used...")

campaign_df["Channel_Used"] = (
    campaign_df["Channel_Used"]
    .fillna("")
    .astype(str)
)

campaign_df["Channel_Used"] = campaign_df[
    "Channel_Used"
].apply(
    lambda x: [
        item.strip().title()
        for item in x.split(",")
        if item.strip()
    ]
)

mlb = MultiLabelBinarizer()

encoded_channels = pd.DataFrame(
    mlb.fit_transform(
        campaign_df["Channel_Used"]
    ),
    columns=[
        f"Channel_{channel}"
        for channel in mlb.classes_
    ],
    index=campaign_df.index
)

campaign_df.drop(
    columns=["Channel_Used"],
    inplace=True
)

campaign_df = pd.concat(
    [
        campaign_df,
        encoded_channels
    ],
    axis=1
)

print("Channel Encoding Completed")

joblib.dump(
    mlb.classes_,
    "channel_classes.pkl"
)

print("channel_classes.pkl Saved")

print("\nEncoded Channel Columns:")
print(
    encoded_channels.columns.tolist()
)

print("\nMissing Values:")
print(
    campaign_df.isnull().sum()
)

print("\nDataset Shape After Feature Engineering:")
print(
    campaign_df.shape
)

print("\nData Types:")
print(
    campaign_df.dtypes
)

print("\nFirst Five Rows:")
print(
    campaign_df.head()
)

campaign_df.to_csv(
    featured_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nFeature Engineered Dataset Saved Successfully")
