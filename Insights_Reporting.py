# INSIGHTS & REPORTING

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

INPUT_FILE = "feature_engineered_marketing_campaign_data.csv"

campaign_df = pd.read_csv(INPUT_FILE)

print("\nDataset Loaded Successfully")
print("Dataset Shape:", campaign_df.shape)

# ==========================================================
# BASIC NUMERIC CONVERSION
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
# HANDLE MISSING NUMERIC VALUES
# ==========================================================

for col in numeric_columns:
    if col in campaign_df.columns:
        median_value = campaign_df[col].median()

        if pd.isna(median_value):
            median_value = 0

        campaign_df[col] = campaign_df[col].fillna(
            median_value
        )

# ==========================================================
# CREATE PROFIT FLAG IF NOT AVAILABLE
# ==========================================================

if "Profit_Flag" not in campaign_df.columns and "ROI" in campaign_df.columns:
    campaign_df["Profit_Flag"] = np.where(
        campaign_df["ROI"] > 0,
        1,
        0
    ).astype(int)

# ==========================================================
# 1. OVERALL CAMPAIGN PERFORMANCE
# ==========================================================

print("\n" + "=" * 70)
print("1. OVERALL CAMPAIGN PERFORMANCE")
print("=" * 70)

total_campaigns = len(campaign_df)

total_revenue = campaign_df["Revenue"].sum()

total_cost = campaign_df["Acquisition_Cost"].sum()

average_revenue = campaign_df["Revenue"].mean()

average_roi = campaign_df["ROI"].mean()

total_clicks = campaign_df["Clicks"].sum()

total_conversions = campaign_df["Conversions"].sum()

average_ctr = campaign_df["CTR"].mean()

average_conversion_rate = campaign_df[
    "Conversion_Rate"
].mean()

profit_campaigns = (
    campaign_df["Profit_Flag"] == 1
).sum()

loss_campaigns = (
    campaign_df["Profit_Flag"] == 0
).sum()

profit_percentage = (
    profit_campaigns / total_campaigns
) * 100

print("Total Campaigns :", total_campaigns)
print("Total Revenue   :", round(total_revenue, 2))
print("Total Cost      :", round(total_cost, 2))
print("Average Revenue :", round(average_revenue, 2))
print("Average ROI     :", round(average_roi, 4))
print("Total Clicks    :", total_clicks)
print("Total Conversions:", total_conversions)
print("Average CTR     :", round(average_ctr, 4))
print(
    "Average Conversion Rate :",
    round(average_conversion_rate, 4)
)
print("Profit Campaigns:", profit_campaigns)
print("Loss Campaigns  :", loss_campaigns)
print(
    "Profit Campaign Percentage :",
    round(profit_percentage, 2),
    "%"
)

# ==========================================================
# 2. BRAND PERFORMANCE
# ==========================================================

print("\n" + "=" * 70)
print("2. BRAND PERFORMANCE INSIGHTS")
print("=" * 70)

if "Brand" in campaign_df.columns:

    brand_analysis = campaign_df.groupby(
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

    brand_analysis = brand_analysis.sort_values(
        "Total_Revenue",
        ascending=False
    )

    print("\nBrand Performance:")
    print(brand_analysis)

    brand_analysis.to_csv(
        "Brand_Insights.csv",
        index=False
    )

    best_brand_revenue = brand_analysis.iloc[0]

    best_brand_roi = brand_analysis.loc[
        brand_analysis["Average_ROI"].idxmax()
    ]

    print(
        "\nHighest Revenue Brand:",
        best_brand_revenue["Brand"]
    )

    print(
        "Highest Revenue:",
        round(
            best_brand_revenue["Total_Revenue"],
            2
        )
    )

    print(
        "\nHighest ROI Brand:",
        best_brand_roi["Brand"]
    )

    print(
        "Highest Average ROI:",
        round(
            best_brand_roi["Average_ROI"],
            4
        )
    )

# ==========================================================
# 3. TOP PERFORMING CAMPAIGNS
# ==========================================================

print("\n" + "=" * 70)
print("3. TOP PERFORMING CAMPAIGNS")
print("=" * 70)

top_revenue_campaigns = campaign_df.nlargest(
    10,
    "Revenue"
)

print("\nTop 10 Revenue Campaigns:")

print(
    top_revenue_campaigns[
        [
            "Campaign_ID",
            "Brand",
            "Revenue",
            "ROI"
        ]
    ]
)

top_revenue_campaigns.to_csv(
    "Top_10_Revenue_Campaigns.csv",
    index=False
)

# ==========================================================
# 4. LOW PERFORMING CAMPAIGNS
# ==========================================================

print("\n" + "=" * 70)
print("4. LOW PERFORMING CAMPAIGNS")
print("=" * 70)

bottom_revenue_campaigns = campaign_df.nsmallest(
    10,
    "Revenue"
)

print("\nBottom 10 Revenue Campaigns:")

print(
    bottom_revenue_campaigns[
        [
            "Campaign_ID",
            "Brand",
            "Revenue",
            "ROI"
        ]
    ]
)

bottom_revenue_campaigns.to_csv(
    "Bottom_10_Revenue_Campaigns.csv",
    index=False
)

# ==========================================================
# 5. ROI INSIGHTS
# ==========================================================

print("\n" + "=" * 70)
print("5. ROI AND PROFITABILITY INSIGHTS")
print("=" * 70)

highest_roi_campaigns = campaign_df.nlargest(
    10,
    "ROI"
)

lowest_roi_campaigns = campaign_df.nsmallest(
    10,
    "ROI"
)

print("\nTop 10 ROI Campaigns:")

print(
    highest_roi_campaigns[
        [
            "Campaign_ID",
            "Brand",
            "Revenue",
            "ROI"
        ]
    ]
)

print("\nBottom 10 ROI Campaigns:")

print(
    lowest_roi_campaigns[
        [
            "Campaign_ID",
            "Brand",
            "Revenue",
            "ROI"
        ]
    ]
)

highest_roi_campaigns.to_csv(
    "Top_10_ROI_Campaigns.csv",
    index=False
)

lowest_roi_campaigns.to_csv(
    "Bottom_10_ROI_Campaigns.csv",
    index=False
)

# ==========================================================
# 6. CHANNEL-WISE INSIGHTS
# ==========================================================

print("\n" + "=" * 70)
print("6. CHANNEL-WISE EFFECTIVENESS")
print("=" * 70)

channel_columns = [
    col for col in campaign_df.columns
    if col.startswith("Channel_")
]

channel_results = []

for channel_col in channel_columns:

    channel_data = campaign_df[
        campaign_df[channel_col] == 1
    ]

    if len(channel_data) > 0:

        channel_results.append({
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
            "Average_Engagement": channel_data[
                "Engagement_Score"
            ].mean()
        })

if channel_results:

    channel_analysis = pd.DataFrame(
        channel_results
    )

    channel_analysis = channel_analysis.sort_values(
        "Total_Revenue",
        ascending=False
    )

    print("\nChannel Performance:")
    print(channel_analysis)

    channel_analysis.to_csv(
        "Channel_Insights.csv",
        index=False
    )

    best_channel_revenue = channel_analysis.iloc[0]

    best_channel_roi = channel_analysis.loc[
        channel_analysis["Average_ROI"].idxmax()
    ]

    print(
        "\nBest Channel by Revenue:",
        best_channel_revenue["Channel"]
    )

    print(
        "Total Revenue:",
        round(
            best_channel_revenue["Total_Revenue"],
            2
        )
    )

    print(
        "\nBest Channel by ROI:",
        best_channel_roi["Channel"]
    )

    print(
        "Average ROI:",
        round(
            best_channel_roi["Average_ROI"],
            4
        )
    )

# ==========================================================
# 7. CORRELATION-BASED KEY FACTORS
# ==========================================================

print("\n" + "=" * 70)
print("7. KEY FACTORS AFFECTING CAMPAIGN PERFORMANCE")
print("=" * 70)

factor_columns = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Acquisition_Cost",
    "ROI",
    "Engagement_Score",
    "CTR",
    "Conversion_Rate",
    "Cost_Per_Click",
    "Cost_Per_Conversion",
    "Lead_Conversion_Rate"
]

available_factor_columns = [
    col for col in factor_columns
    if col in campaign_df.columns
]

correlation_with_revenue = (
    campaign_df[
        available_factor_columns + ["Revenue"]
    ]
    .corr()["Revenue"]
    .drop("Revenue")
    .sort_values(
        ascending=False
    )
)

print(
    "\nCorrelation with Revenue:"
)

print(correlation_with_revenue)

correlation_with_revenue.to_csv(
    "Revenue_Correlation_Insights.csv"
)

# ==========================================================
# 8. PROFITABILITY FACTORS
# ==========================================================

profitability_columns = [
    "Revenue",
    "Acquisition_Cost",
    "Clicks",
    "Leads",
    "Conversions",
    "ROI",
    "Engagement_Score",
    "CTR",
    "Conversion_Rate"
]

available_profitability_columns = [
    col for col in profitability_columns
    if col in campaign_df.columns
]

profitability_correlation = (
    campaign_df[
        available_profitability_columns
    ]
    .corr()["ROI"]
    .drop("ROI")
    .sort_values(
        ascending=False
    )
)

print(
    "\nCorrelation with ROI:"
)

print(profitability_correlation)

profitability_correlation.to_csv(
    "ROI_Correlation_Insights.csv"
)

# ==========================================================
# 9. MODEL PREDICTION SUPPORT
# ==========================================================

print("\n" + "=" * 70)
print("9. MODEL PREDICTION SUPPORT")
print("=" * 70)

regression_model_path = (
    "models/Best_Regression_Model.pkl"
)

classification_model_path = (
    "models/Best_Classification_Model.pkl"
)

try:

    best_regression_model = joblib.load(
        regression_model_path
    )

    print(
        "\nBest Regression Model Loaded Successfully"
    )

except Exception as error:

    best_regression_model = None

    print(
        "\nRegression Model could not be loaded:"
    )

    print(error)

try:

    best_classification_model = joblib.load(
        classification_model_path
    )

    print(
        "Best Classification Model Loaded Successfully"
    )

except Exception as error:

    best_classification_model = None

    print(
        "Classification Model could not be loaded:"
    )

    print(error)

# ==========================================================
# 10. MODEL INFORMATION
# ==========================================================

if best_regression_model is not None:

    print(
        "\nBest Regression Model:"
    )

    print(
        type(
            best_regression_model
        ).__name__
    )

if best_classification_model is not None:

    print(
        "\nBest Classification Model:"
    )

    print(
        type(
            best_classification_model
        ).__name__
    )

# ==========================================================
# 11. DATA-DRIVEN RECOMMENDATIONS
# ==========================================================

print("\n" + "=" * 70)
print("11. DATA-DRIVEN MARKETING RECOMMENDATIONS")
print("=" * 70)

recommendations = []

# Brand recommendation

if "brand_analysis" in locals():

    recommendations.append(
        "Prioritize campaigns for the brand with the highest "
        "total revenue while monitoring ROI to maintain profitability."
    )

    recommendations.append(
        "Use the highest-ROI brand as a benchmark for campaign "
        "planning and budget allocation."
    )

# Channel recommendation

if "channel_analysis" in locals():

    recommendations.append(
        "Increase focus on high-performing channels based on "
        "revenue, ROI, conversions, and engagement."
    )

    recommendations.append(
        "Review low-performing channels and optimize or reduce "
        "spending where ROI remains consistently weak."
    )

# Conversion recommendation

if (
    "Conversion_Rate" in campaign_df.columns
):

    recommendations.append(
        "Improve conversion rates by optimizing campaign content, "
        "target audience selection, landing pages, and calls-to-action."
    )

# Engagement recommendation

if (
    "Engagement_Score" in campaign_df.columns
):

    recommendations.append(
        "Use engagement performance as an indicator when designing "
        "future campaigns and creative strategies."
    )

# Cost recommendation

if (
    "Acquisition_Cost" in campaign_df.columns
    and "ROI" in campaign_df.columns
):

    recommendations.append(
        "Monitor acquisition cost carefully and prioritize campaigns "
        "that generate stronger ROI without excessive spending."
    )

# Profit recommendation

recommendations.append(
    "Use the Profit_Flag prediction to identify campaigns with "
    "higher probability of profitability before allocating budget."
)

# Revenue recommendation

recommendations.append(
    "Use Revenue predictions to estimate expected campaign returns "
    "and support budget allocation decisions."
)

for index, recommendation in enumerate(
    recommendations,
    start=1
):

    print(
        f"{index}. {recommendation}"
    )

# ==========================================================
# SAVE RECOMMENDATIONS
# ==========================================================

recommendation_df = pd.DataFrame({
    "Recommendation": recommendations
})

recommendation_df.to_csv(
    "Marketing_Recommendations.csv",
    index=False
)

# ==========================================================
# 12. CREATE FINAL INSIGHTS REPORT
# ==========================================================

report_lines = []

report_lines.append(
    "MARKETING CAMPAIGN PERFORMANCE - INSIGHTS REPORT"
)

report_lines.append(
    "=" * 60
)

report_lines.append(
    f"Total Campaigns: {total_campaigns}"
)

report_lines.append(
    f"Total Revenue: {total_revenue:.2f}"
)

report_lines.append(
    f"Total Acquisition Cost: {total_cost:.2f}"
)

report_lines.append(
    f"Average ROI: {average_roi:.4f}"
)

report_lines.append(
    f"Profit Campaigns: {profit_campaigns}"
)

report_lines.append(
    f"Loss Campaigns: {loss_campaigns}"
)

report_lines.append(
    f"Profit Campaign Percentage: {profit_percentage:.2f}%"
)

# Brand insight

if "brand_analysis" in locals():

    report_lines.append(
        f"Highest Revenue Brand: "
        f"{best_brand_revenue['Brand']}"
    )

    report_lines.append(
        f"Highest ROI Brand: "
        f"{best_brand_roi['Brand']}"
    )

# Channel insight

if "channel_analysis" in locals():

    report_lines.append(
        f"Best Revenue Channel: "
        f"{best_channel_revenue['Channel']}"
    )

    report_lines.append(
        f"Best ROI Channel: "
        f"{best_channel_roi['Channel']}"
    )

# Key revenue factor

if not correlation_with_revenue.empty:

    highest_revenue_factor = (
        correlation_with_revenue.idxmax()
    )

    report_lines.append(
        f"Strongest Positive Revenue Factor: "
        f"{highest_revenue_factor}"
    )

# Recommendations

report_lines.append(
    "\nDATA-DRIVEN RECOMMENDATIONS"
)

for index, recommendation in enumerate(
    recommendations,
    start=1
):

    report_lines.append(
        f"{index}. {recommendation}"
    )

with open(
    "Marketing_Insights_Report.txt",
    "w",
    encoding="utf-8"
) as report_file:

    report_file.write(
        "\n".join(report_lines)
    )

# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("1. Brand_Insights.csv")
print("2. Top_10_Revenue_Campaigns.csv")
print("3. Bottom_10_Revenue_Campaigns.csv")
print("4. Top_10_ROI_Campaigns.csv")
print("5. Bottom_10_ROI_Campaigns.csv")
print("6. Channel_Insights.csv")
print("7. Revenue_Correlation_Insights.csv")
print("8. ROI_Correlation_Insights.csv")
print("9. Marketing_Recommendations.csv")
print("10. Marketing_Insights_Report.txt")

