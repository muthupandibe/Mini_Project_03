# ==========================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

plt.style.use("ggplot")

# ==========================================================
# LOAD DATASET
# ==========================================================

campaign_df = pd.read_csv(
    "feature_engineered_marketing_campaign_data.csv"
)

print("="*70)
print("STEP 4 : EXPLORATORY DATA ANALYSIS")
print("="*70)

print("\nDataset Shape")
print(campaign_df.shape)

# ==========================================================
# NUMERIC CONVERSION
# ==========================================================

numeric_columns = [

    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "ROI",
    "Engagement_Score",
    "CTR",
    "Conversion_Rate",
    "Cost_Per_Click",
    "Cost_Per_Conversion",
    "Lead_Conversion_Rate"

]

for col in numeric_columns:

    if col in campaign_df.columns:

        campaign_df[col] = pd.to_numeric(
            campaign_df[col],
            errors="coerce"
        )

# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

for col in numeric_columns:

    if col in campaign_df.columns:

        campaign_df[col].fillna(
            campaign_df[col].median(),
            inplace=True
        )

print("\nMissing Values")
print(campaign_df.isnull().sum())

print("\nDataset Information")
campaign_df.info()

print("\nStatistical Summary")
print(campaign_df.describe(include="all"))

# ==========================================================
# CAMPAIGN COUNT BY BRAND
# ==========================================================

if "Brand" in campaign_df.columns:

    plt.figure(figsize=(8,5))

    sns.countplot(
        data=campaign_df,
        x="Brand"
    )

    plt.title("Campaign Count by Brand")

    plt.tight_layout()

    plt.show()

# ==========================================================
# REVENUE BY BRAND
# ==========================================================

brand_revenue = campaign_df.groupby(
    "Brand",
    as_index=False
)["Revenue"].sum()

plt.figure(figsize=(8,5))

sns.barplot(
    data=brand_revenue,
    x="Brand",
    y="Revenue"
)

plt.title("Revenue by Brand")

plt.tight_layout()

plt.show()

# ==========================================================
# ROI BY BRAND
# ==========================================================

brand_roi = campaign_df.groupby(
    "Brand",
    as_index=False
)["ROI"].mean()

plt.figure(figsize=(8,5))

sns.barplot(
    data=brand_roi,
    x="Brand",
    y="ROI"
)

plt.title("Average ROI by Brand")

plt.tight_layout()

plt.show()

# ==========================================================
# CAMPAIGN TYPE
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=campaign_df,
    x="Campaign_Type"
)

plt.xticks(rotation=30)

plt.title("Campaign Type Distribution")

plt.tight_layout()

plt.show()

# ==========================================================
# TARGET AUDIENCE
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=campaign_df,
    x="Target_Audience"
)

plt.xticks(rotation=30)

plt.title("Target Audience Distribution")

plt.tight_layout()

plt.show()

# ==========================================================
# REVENUE DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    campaign_df["Revenue"],
    bins=30,
    kde=True
)

plt.title("Revenue Distribution")

plt.tight_layout()

plt.show()

# ==========================================================
# ROI DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    campaign_df["ROI"],
    bins=30,
    kde=True
)

plt.title("ROI Distribution")

plt.tight_layout()

plt.show()

# ==========================================================
# REVENUE OUTLIERS
# ==========================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x=campaign_df["Revenue"]
)

plt.title("Revenue Outliers")

plt.tight_layout()

plt.show()

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

plt.figure(figsize=(16,10))

sns.heatmap(

    campaign_df.select_dtypes(include=np.number).corr(),

    annot=True,

    cmap="coolwarm",

    fmt=".2f"

)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()

# ==========================================================
# ENGAGEMENT SCORE VS REVENUE
# ==========================================================

plt.figure(figsize=(8,5))

sns.scatterplot(

    data=campaign_df,

    x="Engagement_Score",

    y="Revenue",

    hue="Brand"

)

plt.title("Engagement Score vs Revenue")

plt.tight_layout()

plt.show()

# ==========================================================
# CTR VS REVENUE
# ==========================================================

if "CTR" in campaign_df.columns:

    plt.figure(figsize=(8,5))

    sns.scatterplot(

        data=campaign_df,

        x="CTR",

        y="Revenue",

        hue="Brand"

    )

    plt.title("CTR vs Revenue")

    plt.tight_layout()

    plt.show()

# ==========================================================
# PROFIT FLAG
# ==========================================================

if "Profit_Flag" in campaign_df.columns:

    plt.figure(figsize=(6,4))

    sns.countplot(
        data=campaign_df,
        x="Profit_Flag"
    )

    plt.title("Profit vs Loss Campaigns")

    plt.tight_layout()

    plt.show()

# ==========================================================
# TOP 10 CAMPAIGNS
# ==========================================================

top10 = campaign_df.nlargest(
    10,
    "Revenue"
)

plt.figure(figsize=(12,5))

sns.barplot(

    data=top10,

    x="Campaign_ID",

    y="Revenue"

)

plt.xticks(rotation=45)

plt.title("Top 10 Revenue Campaigns")

plt.tight_layout()

plt.show()

# ==========================================================
# PLOTLY CHART
# ==========================================================

fig = px.bar(

    brand_revenue,

    x="Brand",

    y="Revenue",

    color="Brand",

    title="Brand Revenue Comparison"

)

fig.write_html("brand_revenue.html")

fig.show()

# ==========================================================
# SAVE SUMMARY
# ==========================================================

campaign_df.describe(include="all").to_csv(
    "EDA_Summary.csv"
)

print("\nEDA Summary Saved")

print("brand_revenue.html Saved")
