📊 Multi-Brand Marketing Campaign Performance Prediction

📌 Project Overview

The Multi-Brand Marketing Campaign Performance Prediction project is a Machine Learning based application developed to analyze and predict the performance of marketing campaigns across multiple brands.

The project combines Data Collection, Data Preprocessing, Feature Engineering, Exploratory Data Analysis (EDA), Machine Learning, Model Evaluation, and Streamlit Deployment.

The system provides two major predictions:

💰 Revenue Prediction using Regression

🎯 Profit/Loss Prediction using Classification

The final trained Machine Learning models are integrated into an interactive Streamlit web application.

🎯 Project Objective

The main objective of this project is to help businesses understand marketing campaign performance and make data-driven decisions.

Main Objectives

Predict expected campaign revenue.

Predict whether a campaign will be profitable or result in a loss.

Analyze campaign performance across different brands.

Identify effective marketing channels.

Analyze target audience performance.

Understand important factors affecting campaign revenue.

Generate useful business recommendations.

Provide an interactive prediction dashboard using Streamlit.

🏢 Brands Covered

The project combines marketing campaign information from multiple brands.

Source Brands

Nykaa

Purplle

Tira

A Brand column is added to identify the source brand.

🔄 Project Workflow

Data Collection
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Exploratory Data Analysis
       ↓
Feature Selection
       ↓
Model Building
       ↓
GridSearchCV Hyperparameter Tuning
       ↓
Model Evaluation
       ↓
Insights & Reporting
       ↓
Model Saving
       ↓
Streamlit Deployment

📂 Project Structure

Project_no_03/
│
├── Data_Collection.py
├── Data_Preprocessing.py
├── Feature_Engineering.py
├── Step4_EDA.py
├── Step5_Model_Building.py
├── Step6_Model_Evaluation.py
├── Step7_Insights_Reporting.py
├── app.py
│
├── nykaa_campaign_data_with_nulls.csv
├── purplle_campaign_data_with_nulls.csv
├── tira_campaign_data_with_nulls.csv
│
├── combined_marketing_campaign_data.csv
├── cleaned_marketing_campaign_data.csv
├── feature_engineered_marketing_campaign_data.csv
│
└── models/
    ├── Best_Regression_Model.pkl
    └── Best_Classification_Model.pkl

📥 1. Data Collection

Marketing campaign data is collected from three different brand datasets:

Nykaa

Purplle

Tira

A Brand column is added to each dataset to identify the corresponding brand.

The datasets are then combined into a single dataset.

Output

combined_marketing_campaign_data.csv

🧹 2. Data Preprocessing

The collected dataset is cleaned and prepared for analysis and Machine Learning.

Preprocessing Steps

Handling missing values.

Converting data types.

Removing duplicate records.

Handling numerical variables.

Handling categorical variables.

Checking data consistency.

Preparing clean data for feature engineering.

Missing Value Treatment

For numerical columns:

Median Imputation

For categorical columns:

Mode Imputation

Output

cleaned_marketing_campaign_data.csv

⚙️ 3. Feature Engineering

Feature Engineering is performed to create meaningful features from the existing campaign variables.

CTR

CTR = (Clicks / Impressions) × 100

CTR measures the percentage of impressions that resulted in clicks.

Conversion Rate

Conversion Rate = (Conversions / Clicks) × 100

It measures how effectively clicks are converted into conversions.

Cost Per Click

Cost Per Click = Acquisition Cost / Clicks

Cost Per Conversion

Cost Per Conversion = Acquisition Cost / Conversions

Lead Conversion Rate

Lead Conversion Rate = (Conversions / Leads) × 100

Marketing Channel Encoding

Marketing channels are converted into numerical features using encoding techniques.

The project includes channel features such as:

Channel_Email
Channel_Facebook
Channel_Google
Channel_Instagram
Channel_Whatsapp
Channel_Youtube

🎯 Profit Flag

A binary target variable called Profit_Flag is created based on campaign ROI.

Profit_Flag = 1 → Profit
Profit_Flag = 0 → Loss

This target variable is used for the classification problem.

📊 4. Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the dataset and identify important patterns and relationships.

EDA Techniques

Univariate Analysis

Bivariate Analysis

Multivariate Analysis

Distribution Analysis

Correlation Analysis

Outlier Analysis

Visualizations

The project includes:

Brand-wise Revenue

Campaign Type Performance

Top Performing Campaigns

Lowest Performing Campaigns

Spend vs Revenue

Correlation Heatmap

Marketing Channel Analysis

Target Audience Analysis

Revenue Distribution

Campaign Performance Analysis

🤖 5. Machine Learning

Two Machine Learning problems are implemented.

💰 Regression

The regression model predicts:

Revenue

Regression Algorithms

Linear Regression

Decision Tree Regressor

Random Forest Regressor

The best-performing regression model is selected for final prediction.

🎯 Classification

The classification model predicts:

Profit_Flag

Classification Algorithms

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

The best-performing classification model is selected for final prediction.

🔍 6. Hyperparameter Tuning

GridSearchCV is used for hyperparameter tuning.

Purpose of GridSearchCV

Find the best hyperparameter combination.

Improve model performance.

Reduce manual parameter selection.

Improve model generalization.

Select the best model configuration.

📈 7. Model Evaluation

Regression Evaluation Metrics

R² Score

MAE

MSE

RMSE

MAPE

Classification Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

ROC Curve

ROC-AUC

🏆 Model Performance

Regression Results

The current model evaluation produced approximately:

R²   : 0.92
MAE  : 67,719.23
RMSE : 134,849.96
MAPE : 18.86%

An R² score of approximately 0.92 indicates that the regression model explains a large proportion of the variation in campaign revenue.

Classification Results

The classification model achieved approximately:

Accuracy : 90.33%

An accuracy of approximately 90.33% indicates that the selected classification model performs well in predicting whether a campaign is profitable or not.

Note: Other classification metrics such as Precision, Recall, F1 Score and ROC-AUC should be updated if the final evaluation results are available.

💾 8. Model Saving

The final trained models are saved using joblib.

models/
│
├── Best_Regression_Model.pkl
└── Best_Classification_Model.pkl

The Streamlit application loads these trained models directly.

The final Streamlit application does not require:

regression_features.pkl
classification_features.pkl

🌐 9. Streamlit Application

The Machine Learning models are deployed using Streamlit.

The application provides an interactive interface for campaign analysis and prediction.

🏠 Dashboard

The dashboard provides:

Total Campaigns

Average Revenue

Average ROI

Average CTR

Brand-wise Revenue

Campaign Type Performance

Target Audience Analysis

Marketing Channel Usage

Correlation Heatmap

Dataset Preview

Model Information

Feature Importance

🔮 Prediction

Users can enter:

Campaign Type

Target Audience

Brand

Language

Customer Segment

Marketing Channel

Duration

Impressions

Clicks

Leads

Conversions

Acquisition Cost

Engagement Score

The application calculates derived features such as:

CTR
Conversion Rate
Cost Per Click
Cost Per Conversion
Lead Conversion Rate

The trained models then generate:

💰 Predicted Revenue
🎯 Profit/Loss Prediction
📊 Prediction Confidence

📊 Campaign Analytics

The application provides visual analytics including:

Brand-wise Revenue

Campaign Type Performance

Target Audience Revenue

Marketing Channel Usage

Correlation Heatmap

Campaign Performance

Marketing Funnel

Revenue Prediction

📥 Prediction Report

The application allows users to download prediction results as a CSV file.

The report contains:

Campaign Type

Target Audience

Brand

Marketing Channel

Predicted Revenue

Profit/Loss Prediction

Prediction Confidence

💡 Business Recommendations

If the campaign is predicted as PROFIT

Increase campaign budget.

Continue targeting the selected audience.

Scale successful campaigns.

Use successful marketing channels.

Consider similar campaigns for other brands.

Monitor ROI and conversion rate.

If the campaign is predicted as LOSS

Review campaign strategy.

Reduce acquisition cost.

Improve CTR.

Improve conversion rate.

Test alternative marketing channels.

Re-evaluate the target audience.

Optimize campaign spending.

🛠️ Technologies Used

Programming Language

Python

Data Analysis

Pandas
NumPy

Data Visualization

Matplotlib
Seaborn
Plotly

Machine Learning

Scikit-learn

Model Saving

Joblib

Web Application

Streamlit

Development Environment

Visual Studio Code

📦 Installation

Install the required Python libraries:

pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib

▶️ How to Run the Project

Navigate to the project directory:

cd "C:\Users\ARUNADEVI\Desktop\Data Science\Project_no_03"

Run the Streamlit application:

streamlit run app.py

Open the local URL displayed by Streamlit in your web browser.

📊 Key Project Insights

The project helps identify:

Which brands generate higher revenue.

Which campaign types perform better.

Which marketing channels are frequently used.

Which target audiences generate better results.

The relationship between campaign spending and revenue.

The factors influencing campaign profitability.

The importance of CTR and conversion rate.

The effectiveness of different marketing campaigns.

🚀 Future Enhancements

The project can be further enhanced by adding:

Real-time campaign data.

Cloud deployment.

User authentication.

Automated model retraining.

Advanced feature selection.

XGBoost models.

LightGBM models.

Explainable AI using SHAP.

Real-time database integration.

Campaign budget optimization.

Automated business reports.

Real-time monitoring.

Advanced recommendation systems.

🎓 Learning Outcomes

This project provides practical experience in:

Python Programming

Data Collection

Data Cleaning

Data Preprocessing

Feature Engineering

Exploratory Data Analysis

Data Visualization

Regression

Classification

Feature Selection

Hyperparameter Tuning

GridSearchCV

Model Evaluation

Model Saving

Machine Learning Deployment

Streamlit Application Development

Business Insight Generation

🧠 Project Architecture

                    MARKETING DATA
                         │
                         ▼
                 DATA COLLECTION
                         │
                         ▼
                DATA PREPROCESSING
                         │
                         ▼
                 FEATURE ENGINEERING
                         │
                         ▼
                       EDA
                         │
                         ▼
                  FEATURE SELECTION
                         │
                         ▼
                  MODEL BUILDING
                    /          \
                   /            \
                  ▼              ▼
            REGRESSION       CLASSIFICATION
                │                  │
                ▼                  ▼
          Revenue Prediction   Profit/Loss
                │                  │
                └────────┬─────────┘
                         ▼
                  MODEL EVALUATION
                         │
                         ▼
                  BEST MODEL SAVING
                         │
                         ▼
                  STREAMLIT APP
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
         Dashboard   Prediction   Analytics
             │           │           │
             └───────────┼───────────┘
                         ▼
                BUSINESS INSIGHTS

📌 Project Conclusion

The Multi-Brand Marketing Campaign Performance Prediction project demonstrates how Machine Learning can be used to transform marketing campaign data into meaningful business insights.

The project integrates:

Data Collection
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
EDA
       ↓
Machine Learning
       ↓
Model Evaluation
       ↓
Prediction
       ↓
Business Recommendation

The final Streamlit application provides an interactive platform for predicting campaign revenue and profitability.

The project demonstrates the practical application of Data Science and Machine Learning in the field of marketing analytics.

📌 Project Status

✅ Data Collection
✅ Data Preprocessing
✅ Feature Engineering
✅ Exploratory Data Analysis
✅ Feature Selection
✅ Regression Modeling
✅ Classification Modeling
✅ GridSearchCV
✅ Model Evaluation
✅ Model Saving
✅ Streamlit Deployment
✅ Revenue Prediction
✅ Profit/Loss Prediction
✅ Prediction Confidence
✅ Analytics Dashboard
✅ Business Recommendations
✅ Prediction Report Download

🎉 Project Completed Successfully

Multi-Brand Marketing Campaign Performance Prediction using Machine Learning and Streamlit
