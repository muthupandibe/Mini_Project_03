# Marketing Campaign Performance Prediction

## 1. Project Overview

This project uses Machine Learning to analyze marketing campaign data from **Nykaa, Purplle, and Tira**.

The project has two main goals:

- **Revenue Prediction** using Regression
- **Profit/Loss Prediction** using Classification

The project includes data cleaning, EDA, feature engineering, model building, GridSearchCV tuning, evaluation, business insights, and Streamlit deployment.

---

## 2. Business Problem

Marketing campaigns generate data such as:

- Impressions
- Clicks
- Leads
- Conversions
- Acquisition Cost
- Revenue
- ROI
- Customer Segment
- Campaign Type
- Marketing Channels

The aim is to use this data to understand campaign performance, predict revenue, and identify whether a campaign is likely to be profitable or loss-making.

---

## 3. Project Objectives

1. Combine data from Nykaa, Purplle, and Tira.
2. Clean and preprocess the data.
3. Handle missing values and duplicates.
4. Perform Exploratory Data Analysis (EDA).
5. Process the `Channel_Used` feature.
6. Create the `Profit_Flag` target.
7. Build Revenue prediction models.
8. Build Profit/Loss classification models.
9. Use GridSearchCV for hyperparameter tuning.
10. Evaluate and compare the models.
11. Generate business insights.
12. Deploy the project using Streamlit.

---

## 4. Project Workflow

```text
Raw CSV Files
      ↓
Data Collection
      ↓
Data Cleaning & Preprocessing
      ↓
EDA
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Model Building
      ↓
GridSearchCV
      ↓
Model Evaluation
      ↓
Business Insights
      ↓
Streamlit Deployment
```

---

## 5. Dataset

### Input Files

- `nykaa_campaign_data_with_nulls.csv`
- `purplle_campaign_data_with_nulls.csv`
- `tira_campaign_data_with_nulls.csv`

These files are combined into:

- `combined_marketing_campaign_data.csv`

A `Brand` column identifies each brand.

### Important Features

- Campaign_ID
- Campaign_Type
- Date
- Duration
- Target_Audience
- Customer_Segment
- Language
- Impressions
- Clicks
- Leads
- Conversions
- Engagement_Score
- Acquisition_Cost
- Revenue
- ROI
- Channel_Used

### Created Features

- `Brand`
- `Profit_Flag`
- Multi-label channel features

---

## 6. Data Preprocessing

The following steps are performed:

- Handle missing values
- Remove duplicates
- Correct data types
- Prepare numerical features
- Prepare categorical features
- Apply scaling where required

### Missing Values

- Numerical columns → **Median**
- Categorical columns → **Mode**

Median is used because it is less affected by extreme values.

---

## 7. Exploratory Data Analysis

EDA is performed to understand the data before Machine Learning.

### Analysis Includes

- Revenue distribution
- ROI distribution
- Outlier detection
- Acquisition Cost vs Revenue
- Leads vs Conversions
- Brand-wise performance
- Correlation analysis
- Channel performance

### EDA Outputs

- `EDA_Summary.csv`
- `Revenue_Correlation.csv`
- `Top_10_Campaigns.csv`
- `Bottom_10_Campaigns.csv`
- `brand_revenue.html`

---

## 8. Feature Engineering

### Profit_Flag

The classification target is:

```text
1 → Profit
0 → Loss
```

### Targets

**Regression Target:**

```text
Revenue
```

**Classification Target:**

```text
Profit_Flag
```

### Channel_Used

A campaign can use multiple channels, for example:

```text
Facebook, Instagram
Facebook, Google, YouTube
WhatsApp, Email
```

Therefore, `Channel_Used` is processed using **Multi-Label Encoding**.

---

## 9. Data Leakage Prevention

`ROI` is removed from the classification features because it is directly related to profitability.

This helps prevent **data leakage** and avoids unrealistically high model performance.

---

## 10. Machine Learning Models

### A. Regression — Revenue Prediction

**Target:** `Revenue`

Models:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

### B. Classification — Profit/Loss Prediction

**Target:** `Profit_Flag`

Models:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

---

## 11. GridSearchCV

GridSearchCV is used for hyperparameter tuning.

### Purpose

It tests different parameter combinations using cross-validation and helps find a better model configuration.

```text
Base Model
    ↓
Parameter Grid
    ↓
GridSearchCV
    ↓
Cross-Validation
    ↓
Best Parameters
    ↓
Tuned Model
```

---

## 12. Model Evaluation

### Regression Metrics

- MSE
- RMSE
- MAE
- R²

### Regression Results

| Model | R² |
|---|---:|
| Linear Regression | 0.762 |
| Decision Tree Regressor | 0.826 |
| **Random Forest Regressor** | **0.908** |

**Best Regression Model:** Random Forest Regressor

**R² Score:** 0.908

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score

### Classification Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 92.54% |
| Decision Tree Classifier | 94.80% |
| **Random Forest Classifier** | **96.00%** |

**Best Classification Model:** Random Forest Classifier

**Accuracy:** 96%

---

## 13. Feature Importance

Feature importance is used to understand which features contribute most to model predictions.

Output files:

- `Feature_Importance.csv`
- `Regression_Feature_Importance.csv`

---

## 14. Business Insights

The project focuses on factors such as:

- Customer Segment
- Campaign Type
- Marketing Channel
- Engagement Score
- Leads
- Conversions
- Acquisition Cost

These factors help understand campaign performance and support marketing decisions.

---

## 15. Business Recommendations

1. Invest more in high-performing campaigns.
2. Focus on channels with better engagement and conversions.
3. Improve customer targeting.
4. Reduce spending on low-performing campaigns.
5. Use predicted revenue before major campaign investments.
6. Use Profit/Loss predictions when selecting campaigns.
7. Optimize marketing budget allocation.
8. Monitor campaign performance regularly.

---

## 16. Streamlit Application

The Streamlit application is stored in:

```text
app.py
```

The application:

- Accepts campaign details
- Applies preprocessing
- Predicts Revenue
- Predicts Profit/Loss
- Displays prediction results

Run the application using:

```bash
streamlit run app.py
```

---

## 17. Project Structure

```text
PROJECT_NO_03/
│
├── models/
├── app.py
├── Data_Collection.py
├── Data_Preprocessing.py
├── EDA.py
├── Feature_Engineering.py
├── Model_Building.py
├── Model_Evaluation.py
├── Insights_Reporting.py
│
├── nykaa_campaign_data_with_nulls.csv
├── purplle_campaign_data_with_nulls.csv
├── tira_campaign_data_with_nulls.csv
│
├── combined_marketing_campaign_data.csv
├── cleaned_marketing_campaign_data.csv
├── feature_engineered_marketing_campaign_data.csv
│
├── channel_classes.pkl
├── EDA_Summary.csv
├── Revenue_Correlation.csv
├── Top_10_Campaigns.csv
├── Bottom_10_Campaigns.csv
├── brand_revenue.html
│
├── Regression_Comparison.csv
├── Regression_Model_Comparison.csv
├── Classification_Comparison.csv
├── Classification_Model_Comparison.csv
├── Regression_Feature_Importance.csv
└── Feature_Importance.csv
```

---

## 18. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- GridSearchCV
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib / Pickle

---

## 19. Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scriptsctivate
```

Install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib
```

---

## 20. Future Enhancements

- Advanced Machine Learning models
- Better hyperparameter optimization
- Real-time campaign monitoring
- Marketing budget optimization
- Cloud deployment
- Model monitoring
- Automatic model retraining

---

## 21. Conclusion

This project provides an end-to-end Machine Learning solution for marketing campaign performance prediction.

The project:

- Combines data from three brands
- Cleans and analyzes the data
- Performs feature engineering
- Builds Regression and Classification models
- Uses GridSearchCV
- Evaluates model performance
- Generates business insights
- Provides Streamlit deployment

### Final Results

**Best Regression Model:** Random Forest Regressor  
**R²:** 0.908

**Best Classification Model:** Random Forest Classifier  
**Accuracy:** 96%

---

## 22. Quick Project Summary

| Component | Details |
|---|---|
| Domain | Marketing Analytics |
| Brands | Nykaa, Purplle, Tira |
| Regression Target | Revenue |
| Classification Target | Profit_Flag |
| Channel Processing | Multi-Label Encoding |
| Missing Values | Median / Mode |
| Tuning | GridSearchCV |
| Best Regression | Random Forest Regressor |
| Best R² | 0.908 |
| Best Classification | Random Forest Classifier |
| Best Accuracy | 96% |
| Deployment | Streamlit |
