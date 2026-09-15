# EXPLORATORY DATA ANALYSIS (EDA)
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

plt.style.use("ggplot")

INPUT_FILE = "feature_engineered_marketing_campaign_data.csv"

campaign_df = pd.read_csv(INPUT_FILE)

print("\nDataset Shape:")
print(campaign_df.shape)

# NUMERIC CONVERSION
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

# HANDLE MISSING NUMERIC VALUES

for col in numeric_columns:
    if col in campaign_df.columns:
        median_value = campaign_df[col].median()

        if pd.isna(median_value):
            median_value = 0

        campaign_df[col] = campaign_df[col].fillna(
            median_value
        )

print("\nMissing Values:")
print(campaign_df.isnull().sum())

print("\nDataset Information:")
campaign_df.info()

print("\nStatistical Summary:")
print(campaign_df.describe(include="all"))

# CAMPAIGN COUNT BY BRAND

if "Brand" in campaign_df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=campaign_df, x="Brand")
    plt.title("Campaign Count by Brand")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# REVENUE BY BRAND

if "Brand" in campaign_df.columns and "Revenue" in campaign_df.columns:
    brand_revenue = campaign_df.groupby(
        "Brand",
        as_index=False
    )["Revenue"].sum().sort_values(
        "Revenue",
        ascending=False
    )

    print("\nRevenue by Brand:")
    print(brand_revenue)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=brand_revenue,
        x="Brand",
        y="Revenue"
    )
    plt.title("Revenue by Brand")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# ROI BY BRAND

if "Brand" in campaign_df.columns and "ROI" in campaign_df.columns:
    brand_roi = campaign_df.groupby(
        "Brand",
        as_index=False
    )["ROI"].mean().sort_values(
        "ROI",
        ascending=False
    )

    print("\nAverage ROI by Brand:")
    print(brand_roi)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=brand_roi,
        x="Brand",
        y="ROI"
    )
    plt.title("Average ROI by Brand")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# CAMPAIGN TYPE DISTRIBUTION

if "Campaign_Type" in campaign_df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=campaign_df,
        x="Campaign_Type"
    )
    plt.xticks(rotation=30)
    plt.title("Campaign Type Distribution")
    plt.tight_layout()
    plt.show()

# TARGET AUDIENCE DISTRIBUTION

if "Target_Audience" in campaign_df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=campaign_df,
        x="Target_Audience"
    )
    plt.xticks(rotation=30)
    plt.title("Target Audience Distribution")
    plt.tight_layout()
    plt.show()

# REVENUE DISTRIBUTION

if "Revenue" in campaign_df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(
        campaign_df["Revenue"],
        bins=30,
        kde=True
    )
    plt.title("Revenue Distribution")
    plt.xlabel("Revenue")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# ROI DISTRIBUTION

if "ROI" in campaign_df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(
        campaign_df["ROI"],
        bins=30,
        kde=True
    )
    plt.title("ROI Distribution")
    plt.xlabel("ROI")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# REVENUE OUTLIERS

if "Revenue" in campaign_df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x=campaign_df["Revenue"]
    )
    plt.title("Revenue Outliers")
    plt.tight_layout()
    plt.show()

# ROI OUTLIERS

if "ROI" in campaign_df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x=campaign_df["ROI"]
    )
    plt.title("ROI Outliers")
    plt.tight_layout()
    plt.show()

# CORRELATION HEATMAP

numeric_data = campaign_df.select_dtypes(
    include=np.number
)

if not numeric_data.empty:
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(18, 12))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# ENGAGEMENT SCORE VS REVENUE

if all(
    col in campaign_df.columns
    for col in ["Engagement_Score", "Revenue"]
):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=campaign_df,
        x="Engagement_Score",
        y="Revenue",
        hue="Brand" if "Brand" in campaign_df.columns else None
    )
    plt.title("Engagement Score vs Revenue")
    plt.tight_layout()
    plt.show()

# CTR VS REVENUE

if all(
    col in campaign_df.columns
    for col in ["CTR", "Revenue"]
):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=campaign_df,
        x="CTR",
        y="Revenue",
        hue="Brand" if "Brand" in campaign_df.columns else None
    )
    plt.title("CTR vs Revenue")
    plt.tight_layout()
    plt.show()

# ACQUISITION COST VS REVENUE

if all(
    col in campaign_df.columns
    for col in ["Acquisition_Cost", "Revenue"]
):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=campaign_df,
        x="Acquisition_Cost",
        y="Revenue",
        hue="Brand" if "Brand" in campaign_df.columns else None
    )
    plt.title("Acquisition Cost vs Revenue")
    plt.tight_layout()
    plt.show()

# CLICKS VS REVENUE

if all(
    col in campaign_df.columns
    for col in ["Clicks", "Revenue"]
):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=campaign_df,
        x="Clicks",
        y="Revenue",
        hue="Brand" if "Brand" in campaign_df.columns else None
    )
    plt.title("Clicks vs Revenue")
    plt.tight_layout()
    plt.show()

# ACQUISITION COST VS ROI

if all(
    col in campaign_df.columns
    for col in ["Acquisition_Cost", "ROI"]
):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=campaign_df,
        x="Acquisition_Cost",
        y="ROI",
        hue="Brand" if "Brand" in campaign_df.columns else None
    )
    plt.title("Acquisition Cost vs ROI")
    plt.tight_layout()
    plt.show()

# PROFIT VS LOSS CAMPAIGNS

if "Profit_Flag" in campaign_df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(
        data=campaign_df,
        x="Profit_Flag"
    )
    plt.title("Profit vs Loss Campaigns")
    plt.xlabel("Profit Flag (0 = Loss, 1 = Profit)")
    plt.ylabel("Campaign Count")
    plt.tight_layout()
    plt.show()

    print("\nProfit Flag Distribution:")
    print(campaign_df["Profit_Flag"].value_counts())

# TOP 10 REVENUE CAMPAIGNS

if all(
    col in campaign_df.columns
    for col in ["Campaign_ID", "Revenue"]
):
    top10 = campaign_df.nlargest(
        10,
        "Revenue"
    )

    print("\nTop 10 Revenue Campaigns:")
    print(
        top10[
            [
                "Campaign_ID",
                "Brand",
                "Revenue",
                "ROI"
            ]
        ]
    )

    plt.figure(figsize=(12, 5))
    sns.barplot(
        data=top10,
        x="Campaign_ID",
        y="Revenue"
    )
    plt.xticks(rotation=45)
    plt.title("Top 10 Revenue Campaigns")
    plt.tight_layout()
    plt.show()

# BOTTOM 10 REVENUE CAMPAIGNS

if all(
    col in campaign_df.columns
    for col in ["Campaign_ID", "Revenue"]
):
    bottom10 = campaign_df.nsmallest(
        10,
        "Revenue"
    )

    print("\nBottom 10 Revenue Campaigns:")
    print(
        bottom10[
            [
                "Campaign_ID",
                "Brand",
                "Revenue",
                "ROI"
            ]
        ]
    )

    plt.figure(figsize=(12, 5))
    sns.barplot(
        data=bottom10,
        x="Campaign_ID",
        y="Revenue"
    )
    plt.xticks(rotation=45)
    plt.title("Bottom 10 Revenue Campaigns")
    plt.tight_layout()
    plt.show()

# TOP 10 ROI CAMPAIGNS

if all(
    col in campaign_df.columns
    for col in ["Campaign_ID", "ROI"]
):
    top_roi = campaign_df.nlargest(
        10,
        "ROI"
    )

    print("\nTop 10 ROI Campaigns:")
    print(
        top_roi[
            [
                "Campaign_ID",
                "Brand",
                "Revenue",
                "ROI"
            ]
        ]
    )

# BOTTOM 10 ROI CAMPAIGNS

if all(
    col in campaign_df.columns
    for col in ["Campaign_ID", "ROI"]
):
    bottom_roi = campaign_df.nsmallest(
        10,
        "ROI"
    )

    print("\nBottom 10 ROI Campaigns:")
    print(
        bottom_roi[
            [
                "Campaign_ID",
                "Brand",
                "Revenue",
                "ROI"
            ]
        ]
    )

# CHANNEL-WISE EFFECTIVENESS

channel_columns = [
    col for col in campaign_df.columns
    if col.startswith("Channel_")
]

channel_summary = []

for channel_col in channel_columns:

    channel_data = campaign_df[
        campaign_df[channel_col] == 1
    ]

    if len(channel_data) > 0:

        channel_summary.append({
            "Channel": channel_col.replace(
                "Channel_",
                ""
            ),
            "Campaign_Count": len(channel_data),
            "Total_Revenue": channel_data["Revenue"].sum(),
            "Average_Revenue": channel_data["Revenue"].mean(),
            "Average_ROI": channel_data["ROI"].mean(),
            "Total_Clicks": channel_data["Clicks"].sum(),
            "Total_Conversions": channel_data["Conversions"].sum(),
            "Average_Engagement": channel_data["Engagement_Score"].mean()
        })

if channel_summary:

    channel_analysis = pd.DataFrame(
        channel_summary
    ).sort_values(
        "Total_Revenue",
        ascending=False
    )

    print("\nChannel-Wise Effectiveness:")
    print(channel_analysis)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=channel_analysis,
        x="Channel",
        y="Total_Revenue"
    )

    plt.xticks(rotation=30)
    plt.title("Channel-Wise Total Revenue")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=channel_analysis,
        x="Channel",
        y="Average_ROI"
    )

    plt.xticks(rotation=30)
    plt.title("Channel-Wise Average ROI")
    plt.tight_layout()
    plt.show()

# BRAND PERFORMANCE SUMMARY

if "Brand" in campaign_df.columns:

    brand_summary = campaign_df.groupby(
        "Brand"
    ).agg(
        Campaign_Count=("Campaign_ID", "count"),
        Total_Revenue=("Revenue", "sum"),
        Average_Revenue=("Revenue", "mean"),
        Average_ROI=("ROI", "mean"),
        Total_Clicks=("Clicks", "sum"),
        Total_Conversions=("Conversions", "sum"),
        Average_Engagement=("Engagement_Score", "mean")
    ).reset_index()

    brand_summary = brand_summary.sort_values(
        "Total_Revenue",
        ascending=False
    )

    print("\nBrand Performance Summary:")
    print(brand_summary)

# PLOTLY BRAND REVENUE CHART

if "brand_revenue" in locals():

    fig = px.bar(
        brand_revenue,
        x="Brand",
        y="Revenue",
        color="Brand",
        title="Brand Revenue Comparison"
    )

    fig.update_layout(
        xaxis_title="Brand",
        yaxis_title="Total Revenue",
        template="plotly_white"
    )

    fig.write_html("brand_revenue.html")

    fig.show()


print("\nFinal Dataset Shape:")
print(campaign_df.shape)

print("\nFinal Missing Values:")
print(campaign_df.isnull().sum().sum())

print("\nDuplicate Records:")
print(campaign_df.duplicated().sum())

print("\nAll EDA Analysis Completed Successfully!")
