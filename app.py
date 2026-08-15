# ==========================================================
# MULTI-BRAND MARKETING CAMPAIGN PERFORMANCE PREDICTION
# STREAMLIT APPLICATION
# ==========================================================

import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Marketing Campaign Analytics",
    page_icon="📊",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 20px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #dddddd;
        padding: 15px;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "feature_engineered_marketing_campaign_data.csv"
)

REGRESSION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "Best_Regression_Model.pkl"
)

CLASSIFICATION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "Best_Classification_Model.pkl"
)


# ==========================================================
# CHECK REQUIRED FILES
# ==========================================================

required_files = [
    DATA_PATH,
    REGRESSION_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH
]

missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error(
        "❌ Required project files are missing."
    )

    for file in missing_files:

        st.write(file)

    st.stop()


# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():

    data = pd.read_csv(
        DATA_PATH
    )

    return data


# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    regression = joblib.load(
        REGRESSION_MODEL_PATH
    )

    classification = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    return regression, classification


# ==========================================================
# LOAD PROJECT DATA
# ==========================================================

try:

    df = load_dataset()

    (
        regression_model,
        classification_model
    ) = load_models()

except Exception as error:

    st.error(
        f"❌ Error loading project files: {error}"
    )

    st.stop()


# ==========================================================
# GET MODEL FEATURES
# ==========================================================

# We do NOT require regression_features.pkl
# or classification_features.pkl.

if hasattr(
    regression_model,
    "feature_names_in_"
):

    regression_features = list(
        regression_model.feature_names_in_
    )

else:

    regression_features = None


if hasattr(
    classification_model,
    "feature_names_in_"
):

    classification_features = list(
        classification_model.feature_names_in_
    )

else:

    classification_features = None


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class="main-title">
        📊 Multi-Brand Marketing Campaign Analytics
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
        Machine Learning Based Revenue & Profitability Prediction
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Choose a Page",
    [
        "🏠 Dashboard",
        "🔮 Prediction",
        "ℹ️ About"
    ]
)


# ==========================================================
# ==========================================================
# DASHBOARD PAGE
# ==========================================================
# ==========================================================

if page == "🏠 Dashboard":

    st.header(
        "📈 Campaign Performance Dashboard"
    )


    # ======================================================
    # KPI VALUES
    # ======================================================

    total_campaigns = len(df)


    if "Revenue" in df.columns:

        total_revenue = df[
            "Revenue"
        ].sum()

        average_revenue = df[
            "Revenue"
        ].mean()

    else:

        total_revenue = 0

        average_revenue = 0


    if "ROI" in df.columns:

        average_roi = df[
            "ROI"
        ].mean()

    else:

        average_roi = 0


    if "CTR" in df.columns:

        average_ctr = df[
            "CTR"
        ].mean()

    else:

        average_ctr = 0


    # ======================================================
    # KPI CARDS
    # ======================================================

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "📋 Total Campaigns",
            f"{total_campaigns:,}"
        )


    with col2:

        st.metric(
            "💰 Total Revenue",
            f"₹ {total_revenue:,.0f}"
        )


    with col3:

        st.metric(
            "📊 Average Revenue",
            f"₹ {average_revenue:,.0f}"
        )


    with col4:

        st.metric(
            "📈 Average ROI",
            f"{average_roi:.2f}"
        )


    st.divider()


    # ======================================================
    # BRAND PERFORMANCE
    # ======================================================

    st.subheader(
        "🏆 Brand-wise Revenue"
    )


    if (
        "Brand" in df.columns
        and "Revenue" in df.columns
    ):

        brand_df = (
            df.groupby(
                "Brand"
            )["Revenue"]
            .mean()
            .reset_index()
            .sort_values(
                "Revenue",
                ascending=False
            )
        )


        fig = px.bar(
            brand_df,
            x="Brand",
            y="Revenue",
            text_auto=".2s",
            title="Average Revenue by Brand"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ======================================================
    # CAMPAIGN TYPE + TARGET AUDIENCE
    # ======================================================

    left, right = st.columns(2)


    # ------------------------------------------------------
    # CAMPAIGN TYPE
    # ------------------------------------------------------

    with left:

        st.subheader(
            "🎯 Campaign Type Performance"
        )


        if (
            "Campaign_Type" in df.columns
            and "Revenue" in df.columns
        ):

            campaign_df = (
                df.groupby(
                    "Campaign_Type"
                )["Revenue"]
                .mean()
                .reset_index()
                .sort_values(
                    "Revenue",
                    ascending=False
                )
            )


            fig = px.bar(
                campaign_df,
                x="Campaign_Type",
                y="Revenue",
                text_auto=".2s",
                title="Average Revenue"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # ------------------------------------------------------
    # TARGET AUDIENCE
    # ------------------------------------------------------

    with right:

        st.subheader(
            "👥 Target Audience"
        )


        if (
            "Target_Audience" in df.columns
            and "Revenue" in df.columns
        ):

            audience_df = (
                df.groupby(
                    "Target_Audience"
                )["Revenue"]
                .mean()
                .reset_index()
            )


            fig = px.pie(
                audience_df,
                names="Target_Audience",
                values="Revenue",
                hole=0.45,
                title="Revenue Distribution"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # ======================================================
    # CHANNEL ANALYSIS
    # ======================================================

    st.subheader(
        "📣 Marketing Channel Usage"
    )


    channel_columns = [
        "Channel_Email",
        "Channel_Facebook",
        "Channel_Google",
        "Channel_Instagram",
        "Channel_Whatsapp",
        "Channel_Youtube"
    ]


    channel_names = [
        "Email",
        "Facebook",
        "Google",
        "Instagram",
        "Whatsapp",
        "Youtube"
    ]


    channel_counts = []


    for column in channel_columns:

        if column in df.columns:

            channel_counts.append(
                df[column].sum()
            )

        else:

            channel_counts.append(0)


    channel_df = pd.DataFrame({

        "Channel": channel_names,

        "Campaigns": channel_counts

    })


    fig = px.bar(
        channel_df,
        x="Channel",
        y="Campaigns",
        text_auto=True,
        title="Number of Campaigns by Channel"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# ==========================================================
# PREDICTION PAGE
# ==========================================================
# ==========================================================

elif page == "🔮 Prediction":

    st.header(
        "🔮 Campaign Performance Prediction"
    )


    st.info(
        "Enter campaign details to predict Revenue "
        "and Profit/Loss."
    )


    # ======================================================
    # INPUT SECTION
    # ======================================================

    st.subheader(
        "📝 Campaign Details"
    )


    left, right = st.columns(2)


    # ======================================================
    # LEFT SIDE
    # ======================================================

    with left:


        campaign_type = st.selectbox(
            "Campaign Type",
            sorted(
                df[
                    "Campaign_Type"
                ]
                .dropna()
                .unique()
            )
        )


        target_audience = st.selectbox(
            "Target Audience",
            sorted(
                df[
                    "Target_Audience"
                ]
                .dropna()
                .unique()
            )
        )


        brand = st.selectbox(
            "Brand",
            sorted(
                df[
                    "Brand"
                ]
                .dropna()
                .unique()
            )
        )


        language = st.selectbox(
            "Language",
            sorted(
                df[
                    "Language"
                ]
                .dropna()
                .unique()
            )
        )


        customer_segment = st.selectbox(
            "Customer Segment",
            sorted(
                df[
                    "Customer_Segment"
                ]
                .dropna()
                .unique()
            )
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


    # ======================================================
    # RIGHT SIDE
    # ======================================================

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


    st.divider()


    # ======================================================
    # PREDICT BUTTON
    # ======================================================

    predict_button = st.button(
        "🚀 Predict Campaign Performance",
        use_container_width=True
    )


    # ======================================================
    # WHEN BUTTON CLICKED
    # ======================================================

    if predict_button:

        # ==================================================
        # DERIVED FEATURES
        # ==================================================

        ctr = (
            clicks / impressions * 100
            if impressions > 0
            else 0
        )


        conversion_rate = (
            conversions / clicks * 100
            if clicks > 0
            else 0
        )


        cost_per_click = (
            acquisition_cost / clicks
            if clicks > 0
            else 0
        )


        cost_per_conversion = (
            acquisition_cost / conversions
            if conversions > 0
            else 0
        )


        lead_conversion_rate = (
            conversions / leads * 100
            if leads > 0
            else 0
        )


        # ==================================================
        # CREATE INPUT DICTIONARY
        # ==================================================

        input_dict = {

            "Campaign_Type":
                campaign_type,

            "Target_Audience":
                target_audience,

            "Brand":
                brand,

            "Duration":
                duration,

            "Impressions":
                impressions,

            "Clicks":
                clicks,

            "Leads":
                leads,

            "Conversions":
                conversions,

            "Acquisition_Cost":
                acquisition_cost,

            "Language":
                language,

            "Engagement_Score":
                engagement_score,

            "Customer_Segment":
                customer_segment,

            "CTR":
                ctr,

            "Conversion_Rate":
                conversion_rate,

            "Cost_Per_Click":
                cost_per_click,

            "Cost_Per_Conversion":
                cost_per_conversion,

            "Lead_Conversion_Rate":
                lead_conversion_rate,

            "Channel_Email":
                0,

            "Channel_Facebook":
                0,

            "Channel_Google":
                0,

            "Channel_Instagram":
                0,

            "Channel_Whatsapp":
                0,

            "Channel_Youtube":
                0

        }


        # ==================================================
        # CHANNEL ENCODING
        # ==================================================

        channel_column = (
            "Channel_"
            + selected_channel
        )


        if channel_column in input_dict:

            input_dict[
                channel_column
            ] = 1


        # ==================================================
        # CREATE INPUT DATAFRAME
        # ==================================================

        input_data = pd.DataFrame(
            [input_dict]
        )


        # ==================================================
        # ONE HOT ENCODING
        # ==================================================

        input_encoded = pd.get_dummies(
            input_data
        )


        # ==================================================
        # CHECK MODEL FEATURES
        # ==================================================

        if (
            regression_features is None
            or classification_features is None
        ):

            st.error(
                """
                ❌ The saved models do not contain
                feature_names_in_.

                The model feature information is required
                to correctly prepare the prediction input.

                Please save the trained model together with
                its feature names in Step 5.
                """
            )

            st.stop()

        # ==========================================================
        # REGRESSION INPUT
        # ==========================================================

        regression_input = pd.DataFrame(
            0.0,
            index=[0],
            columns=regression_features,
            dtype=float
        )

        for column in input_encoded.columns:
            if column in regression_features:
                regression_input.loc[0, column] = float(
                    input_encoded.loc[0, column]
                )


        # ==========================================================
        # CLASSIFICATION INPUT
        # ==========================================================

        classification_input = pd.DataFrame(
            0.0,
            index=[0],
            columns=classification_features,
            dtype=float
        )

        for column in input_encoded.columns:
            if column in classification_features:
                classification_input.loc[0, column] = float(
                    input_encoded.loc[0, column]
                )


        # ==========================================================
        # FINAL NUMERIC CONVERSION
        # ==========================================================

        regression_input = (
            regression_input
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype(float)
        )

        classification_input = (
            classification_input
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype(float)
        )

        # ==================================================
        # MODEL PREDICTION
        # ==================================================

        try:

            predicted_revenue = (
                regression_model
                .predict(
                    regression_input
                )[0]
            )


            predicted_profit = (
                classification_model
                .predict(
                    classification_input
                )[0]
            )


            # ==================================================
            # CONFIDENCE
            # ==================================================

            confidence = None


            if hasattr(
                classification_model,
                "predict_proba"
            ):

                confidence = (
                    classification_model
                    .predict_proba(
                        classification_input
                    )[0]
                    .max()
                    * 100
                )


            # ==================================================
            # RESULTS
            # ==================================================

            st.divider()


            st.header(
                "🎯 Prediction Results"
            )


            result1, result2, result3 = (
                st.columns(3)
            )


            # --------------------------------------------------
            # REVENUE
            # --------------------------------------------------

            with result1:

                st.metric(
                    "💰 Predicted Revenue",
                    f"₹ {predicted_revenue:,.2f}"
                )


            # --------------------------------------------------
            # PROFIT / LOSS
            # --------------------------------------------------

            with result2:

                if predicted_profit == 1:

                    st.success(
                        "🟢 PROFIT"
                    )

                else:

                    st.error(
                        "🔴 LOSS"
                    )


            # --------------------------------------------------
            # CONFIDENCE
            # --------------------------------------------------

            with result3:

                if confidence is not None:

                    st.metric(
                        "🎯 Confidence",
                        f"{confidence:.2f}%"
                    )

                else:

                    st.metric(
                        "🎯 Confidence",
                        "N/A"
                    )


            # ==================================================
            # CAMPAIGN METRICS
            # ==================================================

            st.subheader(
                "📊 Campaign Performance"
            )


            metric_df = pd.DataFrame({

                "Metric": [
                    "Impressions",
                    "Clicks",
                    "Leads",
                    "Conversions"
                ],

                "Value": [
                    impressions,
                    clicks,
                    leads,
                    conversions
                ]

            })


            fig = px.bar(
                metric_df,
                x="Metric",
                y="Value",
                text_auto=True,
                title="Campaign Funnel Metrics"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


            # ==================================================
            # MARKETING FUNNEL
            # ==================================================

            st.subheader(
                "🔻 Marketing Funnel"
            )


            funnel_df = pd.DataFrame({

                "Stage": [
                    "Impressions",
                    "Clicks",
                    "Leads",
                    "Conversions"
                ],

                "Count": [
                    impressions,
                    clicks,
                    leads,
                    conversions
                ]

            })


            fig = px.funnel(
                funnel_df,
                x="Count",
                y="Stage",
                title="Campaign Conversion Funnel"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


            # ==================================================
            # CAMPAIGN SUMMARY
            # ==================================================

            st.subheader(
                "📋 Campaign Summary"
            )


            summary_df = pd.DataFrame({

                "Feature": [

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

                "Value": [

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

                    round(
                        ctr,
                        2
                    ),

                    round(
                        conversion_rate,
                        2
                    ),

                    round(
                        cost_per_click,
                        2
                    ),

                    round(
                        cost_per_conversion,
                        2
                    ),

                    round(
                        lead_conversion_rate,
                        2
                    )

                ]

            })


            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )


            # ==================================================
            # BUSINESS RECOMMENDATION
            # ==================================================

            st.subheader(
                "💡 Business Recommendation"
            )


            if predicted_profit == 1:

                st.success(
                    """
                    **Recommended Action**

                    • Continue the campaign strategy.

                    • Consider increasing the marketing budget.

                    • Focus on the selected target audience.

                    • Scale successful campaigns to other brands.

                    • Continue monitoring ROI and conversion rate.
                    """
                )

            else:

                st.warning(
                    """
                    **Recommended Action**

                    • Review the campaign strategy.

                    • Reduce acquisition cost.

                    • Improve CTR and conversion rate.

                    • Test alternative marketing channels.

                    • Re-evaluate the target audience.
                    """
                )


            # ==================================================
            # DOWNLOAD REPORT
            # ==================================================

            result_df = pd.DataFrame({

                "Campaign Type": [
                    campaign_type
                ],

                "Target Audience": [
                    target_audience
                ],

                "Brand": [
                    brand
                ],

                "Channel": [
                    selected_channel
                ],

                "Predicted Revenue": [
                    predicted_revenue
                ],

                "Prediction": [

                    "Profit"
                    if predicted_profit == 1
                    else "Loss"

                ],

                "Confidence": [
                    confidence
                ]

            })


            csv_data = result_df.to_csv(
                index=False
            )


            st.download_button(
                label="📥 Download Prediction Report",
                data=csv_data,
                file_name=(
                    "Marketing_Campaign_Prediction.csv"
                ),
                mime="text/csv"
            )


        except Exception as error:

            st.error(
                f"❌ Prediction failed: {error}"
            )


# ==========================================================
# ==========================================================
# ABOUT PAGE
# ==========================================================
# ==========================================================

elif page == "ℹ️ About":

    st.header(
        "ℹ️ About the Project"
    )


    st.markdown(
        """
## 📊 Multi-Brand Marketing Campaign Performance Prediction

### 🎯 Project Objective

The objective of this project is to use Machine Learning
to analyze marketing campaigns and predict their performance.

### 💰 Regression

The regression model predicts:

**Campaign Revenue**

Models evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### 🎯 Classification

The classification model predicts:

**Profit / Loss**

Models evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### 📈 Evaluation Metrics

**Regression**

- R² Score
- MAE
- MSE
- RMSE
- MAPE

**Classification**

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

### 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Joblib

### 💡 Business Value

The application helps businesses:

- Identify potentially profitable campaigns.
- Estimate expected campaign revenue.
- Analyze marketing channels.
- Optimize campaign budgets.
- Improve marketing performance.
        """
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "📊 Marketing Campaign Performance Prediction | "
    "Machine Learning Mini Project"
)
