# ==========================================================
# Streamlit Application
# ==========================================================
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(

    page_title="Marketing Campaign Performance Prediction",

    page_icon="📊",

    layout="wide"

)

# ==========================================================
# PROJECT TITLE
# ==========================================================

st.title("📊 Multi-Brand Marketing Campaign Performance Prediction")

st.markdown("""
Machine Learning models Prediction

- 💰 Revenue (Regression)
- 📈 Profit / Loss (Classification)

""")

st.markdown("---")

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_PATH = os.path.join(
    BASE_DIR,
    "feature_engineered_marketing_campaign_data.csv"
)

# ==========================================================
# CHECK DATASET
# ==========================================================

if not os.path.exists(DATA_PATH):

    st.error("feature_engineered_marketing_campaign_data.csv not found.")

    st.stop()

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(DATA_PATH)

# ==========================================================
# CHECK MODEL FILES
# ==========================================================

required_files = [

    "Best_Regression_Model.pkl",

    "Best_Classification_Model.pkl",

    "regression_features.pkl",

    "classification_features.pkl"

]

missing = []

for file in required_files:

    if not os.path.exists(os.path.join(MODEL_DIR, file)):

        missing.append(file)

if len(missing) > 0:

    st.error("Missing Model Files")

    st.write(missing)

    st.stop()

# ==========================================================
# LOAD MODELS
# ==========================================================

regression_model = joblib.load(

    os.path.join(

        MODEL_DIR,

        "Best_Regression_Model.pkl"

    )

)

classification_model = joblib.load(

    os.path.join(

        MODEL_DIR,

        "Best_Classification_Model.pkl"

    )

)

regression_features = joblib.load(

    os.path.join(

        MODEL_DIR,

        "regression_features.pkl"

    )

)

classification_features = joblib.load(

    os.path.join(

        MODEL_DIR,

        "classification_features.pkl"

    )

)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "Prediction",

        "Dataset Summary",

        "About"

    ]

)

# ==========================================================
# DATASET SUMMARY
# ==========================================================

if page == "Dataset Summary":

    st.header("Dataset Summary")

    st.write("Shape :", df.shape)

    st.dataframe(df.head())

    st.subheader("Columns")

    st.write(df.columns.tolist())

    st.stop()

# ==========================================================
# ABOUT
# ==========================================================

if page == "About":

    st.header("Project Information")

    st.markdown("""

### Project

Multi-Brand Marketing Campaign Performance Prediction

### Regression Models

- Linear Regression

- Decision Tree Regressor

- Random Forest Regressor

### Classification Models

- Logistic Regression

- Decision Tree Classifier

- Random Forest Classifier

### Outputs

- Revenue Prediction

- Profit / Loss Prediction

""")

    st.stop()

# ==========================================================
# PREDICTION PAGE
# ==========================================================

st.header("Campaign Details")

left, right = st.columns(2)

# ==========================================================
# INPUT FORM
# ==========================================================

with left:

    campaign_type = st.selectbox(

        "Campaign Type",

        sorted(df["Campaign_Type"].dropna().unique())

    )

    target_audience = st.selectbox(

        "Target Audience",

        sorted(df["Target_Audience"].dropna().unique())

    )

    brand = st.selectbox(

        "Brand",

        sorted(df["Brand"].dropna().unique())

    )

    language = st.selectbox(

        "Language",

        sorted(df["Language"].dropna().unique())

    )

    customer_segment = st.selectbox(

        "Customer Segment",

        sorted(df["Customer_Segment"].dropna().unique())

    )

    selected_channel = st.selectbox(

        "Marketing Channel",

        [

            "Email",

            "Facebook",

            "Google",

            "Instagram",

            "Whatsapp",

            "Youtube"

        ]

    )

with right:

    duration = st.number_input(

        "Duration (Days)",

        min_value=1,

        max_value=365,

        value=30

    )

    impressions = st.number_input(

        "Impressions",

        min_value=0,

        value=10000

    )

    clicks = st.number_input(

        "Clicks",

        min_value=0,

        value=500

    )

    leads = st.number_input(

        "Leads",

        min_value=0,

        value=100

    )

    conversions = st.number_input(

        "Conversions",

        min_value=0,

        value=50

    )

    acquisition_cost = st.number_input(

        "Acquisition Cost",

        min_value=0.0,

        value=10000.0

    )

    engagement_score = st.number_input(

        "Engagement Score",

        min_value=0.0,

        max_value=100.0,

        value=50.0

    )

# ==========================================================
# PREDICT BUTTON
# ==========================================================

st.markdown("---")

predict_button = st.button(

    "Predict Campaign Performance",

    use_container_width=True

)

# ==========================================================
# CREATE INPUT DATA
# ==========================================================

if predict_button:

    # -----------------------------
    # Calculate Derived Features
    # -----------------------------

    ctr = (clicks / impressions * 100) if impressions > 0 else 0

    conversion_rate = (conversions / clicks * 100) if clicks > 0 else 0

    cost_per_click = (acquisition_cost / clicks) if clicks > 0 else 0

    cost_per_conversion = (acquisition_cost / conversions) if conversions > 0 else 0

    lead_conversion_rate = (conversions / leads * 100) if leads > 0 else 0

    # -----------------------------
    # Input Dictionary
    # -----------------------------

    input_dict = {

        "Campaign_Type": campaign_type,

        "Target_Audience": target_audience,

        "Brand": brand,

        "Duration": duration,

        "Impressions": impressions,

        "Clicks": clicks,

        "Leads": leads,

        "Conversions": conversions,

        "Acquisition_Cost": acquisition_cost,

        "Language": language,

        "Engagement_Score": engagement_score,

        "Customer_Segment": customer_segment,

        "CTR": ctr,

        "Conversion_Rate": conversion_rate,

        "Cost_Per_Click": cost_per_click,

        "Cost_Per_Conversion": cost_per_conversion,

        "Lead_Conversion_Rate": lead_conversion_rate,

        "Channel_Email": 0,

        "Channel_Facebook": 0,

        "Channel_Google": 0,

        "Channel_Instagram": 0,

        "Channel_Whatsapp": 0,

        "Channel_Youtube": 0

    }

    # -----------------------------
    # Encode Selected Channel
    # -----------------------------

    channel_column = f"Channel_{selected_channel}"

    if channel_column in input_dict:

        input_dict[channel_column] = 1

    # -----------------------------
    # Create DataFrame
    # -----------------------------

    input_data = pd.DataFrame([input_dict])

    st.subheader("Input Data")

    st.dataframe(input_data)

# ==========================================================
# PREPROCESS INPUT DATA
# ==========================================================

    # One-Hot Encoding
    input_encoded = pd.get_dummies(input_data)

    # ======================================================
    # ALIGN REGRESSION FEATURES
    # ======================================================

    regression_input = pd.DataFrame(
        columns=regression_features,
        index=[0]
    )

    regression_input = regression_input.fillna(0)

    for col in input_encoded.columns:

        if col in regression_features:

            regression_input.loc[0, col] = input_encoded.loc[0, col]

    # ======================================================
    # ALIGN CLASSIFICATION FEATURES
    # ======================================================

    classification_input = pd.DataFrame(
        columns=classification_features,
        index=[0]
    )

    classification_input = classification_input.fillna(0)

    for col in input_encoded.columns:

        if col in classification_features:

            classification_input.loc[0, col] = input_encoded.loc[0, col]

    # ======================================================
    # CONVERT NUMERIC COLUMNS
    # ======================================================

    regression_input = regression_input.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)

    classification_input = classification_input.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)

    # ======================================================
    # DISPLAY MODEL INPUT (Optional)
    # ======================================================

    with st.expander("Model Input Features"):

        st.dataframe(regression_input)

    # ======================================================
    # REVENUE PREDICTION
    # ======================================================

    predicted_revenue = regression_model.predict(
        regression_input
    )[0]

    # ======================================================
    # PROFIT / LOSS PREDICTION
    # ======================================================

    predicted_profit = classification_model.predict(
        classification_input
    )[0]

    # ======================================================
    # PREDICTION CONFIDENCE
    # ======================================================

    confidence = None

    if hasattr(classification_model, "predict_proba"):

        confidence = (
            classification_model.predict_proba(
                classification_input
            )[0].max() * 100
        )

    # ======================================================
    # CREATE RESULT DICTIONARY
    # ======================================================

    prediction_result = {

        "Predicted Revenue": float(predicted_revenue),

        "Prediction": (
            "Profit"
            if predicted_profit == 1
            else "Loss"
        ),

        "Confidence": confidence

    }

# ==========================================================
# DISPLAY PREDICTION RESULTS
# ==========================================================

    st.markdown("---")
    st.header("Prediction Results")

    # ======================================================
    # KPI CARDS
    # ======================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="💰 Predicted Revenue",
            value=f"₹ {predicted_revenue:,.2f}"
        )

    with col2:
        if predicted_profit == 1:
            st.success("✅ Campaign Status : PROFIT")
        else:
            st.error("❌ Campaign Status : LOSS")

    with col3:
        if confidence is not None:
            st.metric(
                label="🎯 Confidence",
                value=f"{confidence:.2f}%"
            )
        else:
            st.metric(
                label="🎯 Confidence",
                value="N/A"
            )

    st.markdown("---")

    # ======================================================
    # CAMPAIGN SUMMARY
    # ======================================================

    st.subheader("Campaign Summary")

    summary_df = pd.DataFrame({

        "Feature":[
            "Campaign Type",
            "Target Audience",
            "Brand",
            "Marketing Channel",
            "Duration",
            "Impressions",
            "Clicks",
            "Leads",
            "Conversions",
            "Acquisition Cost",
            "Language",
            "Customer Segment",
            "Engagement Score",
            "CTR (%)",
            "Conversion Rate (%)",
            "Cost Per Click",
            "Cost Per Conversion",
            "Lead Conversion Rate (%)"
        ],

        "Value":[
            campaign_type,
            target_audience,
            brand,
            selected_channel,
            duration,
            impressions,
            clicks,
            leads,
            conversions,
            acquisition_cost,
            language,
            customer_segment,
            engagement_score,
            round(ctr,2),
            round(conversion_rate,2),
            round(cost_per_click,2),
            round(cost_per_conversion,2),
            round(lead_conversion_rate,2)
        ]

    })

    st.dataframe(summary_df, use_container_width=True)

    # ======================================================
    # REVENUE BAR CHART
    # ======================================================

    st.subheader("Revenue Prediction")

    revenue_df = pd.DataFrame({

        "Category":["Predicted Revenue"],

        "Revenue":[predicted_revenue]

    })

    fig1 = px.bar(

        revenue_df,

        x="Category",

        y="Revenue",

        text="Revenue",

        title="Predicted Revenue"

    )

    fig1.update_traces(texttemplate="₹ %{y:,.0f}")

    st.plotly_chart(fig1, width="stretch")

    # ======================================================
    # CAMPAIGN PERFORMANCE
    # ======================================================

    chart_df = pd.DataFrame({

        "Metric":[

            "Impressions",

            "Clicks",

            "Leads",

            "Conversions"

        ],

        "Value":[

            impressions,

            clicks,

            leads,

            conversions

        ]

    })

    fig2 = px.bar(

        chart_df,

        x="Metric",

        y="Value",

        text="Value",

        title="Campaign Performance"

    )

    fig2.update_traces(textposition="outside")

    st.plotly_chart(fig2, width="stretch")

    # ======================================================
    # CONVERSION FUNNEL
    # ======================================================

    funnel_df = pd.DataFrame({

        "Stage":[

            "Impressions",

            "Clicks",

            "Leads",

            "Conversions"

        ],

        "Count":[

            impressions,

            clicks,

            leads,

            conversions

        ]

    })

    fig3 = px.funnel(

        funnel_df,

        x="Count",

        y="Stage",

        title="Marketing Funnel"

    )

    st.plotly_chart(fig3, width="stretch")

    # ======================================================
    # PIE CHART
    # ======================================================

    pie_df = pd.DataFrame({

        "Stage":[

            "Clicks",

            "Leads",

            "Conversions"

        ],

        "Count":[

            clicks,

            leads,

            conversions

        ]

    })

    fig4 = px.pie(

        pie_df,

        names="Stage",

        values="Count",

        hole=0.45,

        title="Campaign Distribution"

    )

    st.plotly_chart(fig4, width="stretch")

    # ======================================================
    # DOWNLOAD RESULTS
    # ======================================================

    result_df = pd.DataFrame({

        "Campaign Type":[campaign_type],
        "Target Audience":[target_audience],
        "Brand":[brand],
        "Channel":[selected_channel],
        "Predicted Revenue":[predicted_revenue],
        "Prediction":[
            "Profit" if predicted_profit==1 else "Loss"
        ],
        "Confidence":[confidence]

    })

    csv = result_df.to_csv(index=False)

    st.download_button(

        label="📥 Download Prediction Report",

        data=csv,

        file_name="Prediction_Report.csv",

        mime="text/csv"

    )

    # ======================================================
    # BUSINESS RECOMMENDATIONS
    # ======================================================

    st.markdown("---")

    st.subheader("Business Recommendations")

    if predicted_profit == 1:

        st.success("""
- Increase budget for similar campaigns.
- Continue targeting this audience.
- Use this marketing channel in future campaigns.
- Scale this campaign across other brands.
""")

    else:

        st.warning("""
- Review campaign strategy.
- Reduce acquisition cost.
- Improve CTR and conversion rate.
- Consider testing a different marketing channel.
""")

# ==========================================================
# ANALYTICS DASHBOARD
# ==========================================================
if page == "Dataset Summary":

    st.markdown("---")

st.header("📈 Campaign Analytics Dashboard")

# ==========================================================
# KPI SUMMARY
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Campaigns",
        len(df)
    )

with col2:
    st.metric(
        "Average Revenue",
        f"₹ {df['Revenue'].mean():,.0f}"
    )

with col3:
    st.metric(
        "Average ROI",
        f"{df['ROI'].mean():.2f}"
    )

with col4:
    st.metric(
        "Average CTR",
        f"{df['CTR'].mean():.2f}%"
    )

# ==========================================================
# BRAND ANALYSIS
# ==========================================================

st.subheader("🏆 Brand-wise Revenue")

brand_df = (
    df.groupby("Brand")["Revenue"]
    .mean()
    .reset_index()
)

fig = px.bar(
    brand_df,
    x="Brand",
    y="Revenue",
    color="Brand",
    text_auto=".2s",
    title="Average Revenue by Brand"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# CAMPAIGN TYPE ANALYSIS
# ==========================================================

st.subheader("Campaign Type Performance")

campaign_df = (
    df.groupby("Campaign_Type")["Revenue"]
    .mean()
    .reset_index()
)

fig = px.bar(
    campaign_df,
    x="Campaign_Type",
    y="Revenue",
    color="Campaign_Type",
    text_auto=".2s"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# TARGET AUDIENCE
# ==========================================================

st.subheader("Target Audience Revenue")

aud_df = (
    df.groupby("Target_Audience")["Revenue"]
    .mean()
    .reset_index()
)

fig = px.pie(
    aud_df,
    names="Target_Audience",
    values="Revenue",
    hole=0.45
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# CHANNEL ANALYSIS
# ==========================================================

st.subheader("Marketing Channel Usage")

channel_columns = [

    "Channel_Email",
    "Channel_Facebook",
    "Channel_Google",
    "Channel_Instagram",
    "Channel_Whatsapp",
    "Channel_Youtube"

]

channel_counts = []

for c in channel_columns:

    if c in df.columns:

        channel_counts.append(df[c].sum())

    else:

        channel_counts.append(0)

channel_df = pd.DataFrame({

    "Channel":[
        "Email",
        "Facebook",
        "Google",
        "Instagram",
        "Whatsapp",
        "Youtube"
    ],

    "Campaigns":channel_counts

})

fig = px.bar(

    channel_df,

    x="Channel",

    y="Campaigns",

    color="Channel",

    text_auto=True

)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# CORRELATION MATRIX
# ==========================================================

st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=np.number)

corr = numeric_df.corr()

fig = px.imshow(

    corr,

    text_auto=".2f",

    aspect="auto",

    color_continuous_scale="Viridis"

)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# DATA PREVIEW
# ==========================================================

st.subheader("Dataset Preview")

st.dataframe(df.head(20), use_container_width=True)

# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.subheader("Model Information")

info = pd.DataFrame({

    "Task":[
        "Regression",
        "Classification"
    ],

    "Model":[
        type(regression_model).__name__,
        type(classification_model).__name__
    ]

})

st.table(info)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

if hasattr(regression_model, "feature_importances_"):

    st.subheader("Regression Feature Importance")

    importance = pd.DataFrame({

        "Feature": regression_features,

        "Importance": regression_model.feature_importances_

    })

    importance = importance.sort_values(

        "Importance",

        ascending=False

    ).head(10)

    fig = px.bar(

        importance,

        x="Importance",

        y="Feature",

        orientation="h",

        title="Top 10 Important Features"

    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Marketing Campaign Performance Prediction using Machine Learning & Streamlit"
)