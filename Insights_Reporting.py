# ==========================================================
# INSIGHTS & REPORTING
# ==========================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "feature_engineered_marketing_campaign_data.csv")

df = pd.read_csv(DATA)

print("="*60)
print("STEP 7 : INSIGHTS & REPORTING")
print("="*60)

# ==========================================================
# BASIC SUMMARY
# ==========================================================
print("\nDataset Shape:", df.shape)
print(df.describe(include="all"))

# ==========================================================
# CREATE PROFIT FLAG IF MISSING
# ==========================================================
if "Profit_Flag" not in df.columns:
    df["Profit_Flag"] = ((df["Revenue"]-df["Acquisition_Cost"])>0).astype(int)

# ==========================================================
# BRAND PERFORMANCE
# ==========================================================
if "Brand" in df.columns:
    brand = df.groupby("Brand")[["Revenue","Acquisition_Cost"]].mean().sort_values("Revenue",ascending=False)
    print("\nBrand Performance")
    print(brand)
    brand.plot(kind="bar", figsize=(8,5))
    plt.title("Average Revenue by Brand")
    plt.tight_layout()
    plt.show()

# ==========================================================
# CHANNEL PERFORMANCE
# ==========================================================
if "Channel_Used" in df.columns:
    ch = df.groupby("Channel_Used")["Revenue"].mean().sort_values(ascending=False)
    print("\nChannel Performance")
    print(ch)
    ch.plot(kind="bar", figsize=(8,5))
    plt.title("Average Revenue by Channel")
    plt.tight_layout()
    plt.show()

# ==========================================================
# TOP / BOTTOM CAMPAIGNS
# ==========================================================
if "Campaign_ID" in df.columns:
    top = df.nlargest(10,"Revenue")[["Campaign_ID","Revenue"]]
    bottom = df.nsmallest(10,"Revenue")[["Campaign_ID","Revenue"]]
else:
    top = df.nlargest(10,"Revenue")[["Revenue"]]
    bottom = df.nsmallest(10,"Revenue")[["Revenue"]]

print("\nTop 10 Campaigns")
print(top)
print("\nBottom 10 Campaigns")
print(bottom)

# ==========================================================
# PROFITABILITY
# ==========================================================
profit_rate = df["Profit_Flag"].mean()*100
print(f"\nProfitable Campaigns: {profit_rate:.2f}%")

# ==========================================================
# CORRELATION
# ==========================================================
corr = df.select_dtypes("number").corr()["Revenue"].sort_values(ascending=False)
print("\nFeatures correlated with Revenue")
print(corr)

# ==========================================================
# SAVE REPORTS
# ==========================================================
top.to_csv(os.path.join(BASE_DIR,"Top_10_Campaigns.csv"),index=False)
bottom.to_csv(os.path.join(BASE_DIR,"Bottom_10_Campaigns.csv"),index=False)
corr.to_csv(os.path.join(BASE_DIR,"Revenue_Correlation.csv"))

# ==========================================================
# BUSINESS INSIGHTS
# ==========================================================
print("\nBUSINESS INSIGHTS")
print("- Focus on channels with highest average revenue.")
print("- Increase budget for consistently profitable campaigns.")
print("- Review low-performing campaigns for optimization.")
print("- Improve conversion rates to increase revenue.")
print("- Reduce acquisition cost where ROI is low.")