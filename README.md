# Multi-Brand Marketing Campaign Performance Analysis & Prediction

## 📌 Project Overview

**Marketing Campaign Performance Prediction** is an end-to-end Machine Learning project in the **Marketing Analytics & Business Intelligence** domain.

The project combines campaign datasets from **Nykaa, Purplle, and Tira**, cleans and transforms the raw data, performs Exploratory Data Analysis (EDA), engineers predictive features, builds regression and classification models, evaluates model performance, generates business insights, and deploys predictions through a **Streamlit** application.

The two primary prediction tasks are:

- **Regression:** Predict campaign **Revenue**
- **Classification:** Predict campaign **Profit/Loss** using **Profit_Flag**

---

## 🎯 Business Problem

Marketing teams generate large volumes of campaign data containing impressions, clicks, leads, conversions, acquisition cost, revenue, ROI, customer segments, and marketing channels.

Raw campaign data can contain missing values, duplicate records, inconsistent data types, and a `Channel_Used` field containing multiple channels in a single cell. These issues make analysis and Machine Learning more difficult.

This project transforms the raw campaign data into a structured analytical dataset and uses Machine Learning to support campaign performance evaluation, revenue prediction, profitability prediction, and marketing decision-making.

---

## 🎯 Objectives

1. Combine campaign data from multiple brands into a single dataset.
2. Clean and preprocess raw marketing campaign data.
3. Handle missing values and duplicate records.
4. Correct and standardize data types, including the `Date` field.
5. Apply multi-label encoding to the `Channel_Used` feature.
6. Create `Profit_Flag` from ROI for binary profit/loss classification.
7. Perform EDA to understand campaign, brand, channel, and performance patterns.
8. Build regression models to predict Revenue.
9. Build classification models to predict Profit/Loss.
10. Evaluate and compare different Machine Learning algorithms.
11. Prevent data leakage by excluding ROI from classification features.
12. Generate business insights and recommendations.
13. Deploy the trained models through a Streamlit application.

---

## 🗂️ Dataset

Three raw CSV files were provided:

| Brand | Rows | Columns | Duplicate Rows |
|---|---:|---:|---:|
| Nykaa | 55,555 | 16 | 0 |
| Purplle | 55,555 | 16 | 0 |
| Tira | 55,555 | 16 | 0 |
| **Combined** | **166,665** | **17*** | **0** |

\* The combined dataset contains the original 16 columns plus the engineered `Brand` column used to identify the source brand.

### Raw Dataset Files

- `nykaa_campaign_data_with_nulls.csv`
- `purplle_campaign_data_with_nulls.csv`
- `tira_campaign_data_with_nulls.csv`

### Features

#### Campaign Features
- `Campaign_ID`
- `Campaign_Type`
- `Date`
- `Duration`

#### Customer Features
- `Target_Audience`
- `Customer_Segment`
- `Language`

#### Marketing Performance Features
- `Impressions`
- `Clicks`
- `Leads`
- `Conversions`
- `Engagement_Score`

#### Cost & Financial Features
- `Acquisition_Cost`
- `Revenue`
- `ROI`

#### Marketing Channel Feature
- `Channel_Used`

#### Engineered Feature
- `Brand` — added while combining the three brand datasets
- `Profit_Flag` — created during feature engineering

---

## 🔄 End-to-End Workflow

```text
Raw CSV Files
     ↓
Data Collection
     ↓
CSV → Pandas DataFrame
     ↓
Brand Identification & Dataset Combination
     ↓
Data Cleaning & Preprocessing
     ↓
EDA
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Regression + Classification
     ↓
Model Evaluation & Comparison
     ↓
Insights & Business Recommendations
     ↓
Streamlit Deployment
```

---

# 1. Data Collection

The three brand-specific CSV files are loaded into Pandas DataFrames.

A `Brand` column is added to identify whether each record belongs to Nykaa, Purplle, or Tira, and the datasets are then combined into one analytical dataset.

---

# 2. Data Cleaning & Preprocessing

### Missing Value Handling

- **Numerical features:** Median imputation
- **Categorical features:** Most-frequent-value (mode) imputation

This approach keeps the dataset usable while reducing the effect of extreme numerical observations.

### Duplicate Checking

Duplicate records are checked before model development.

The uploaded raw datasets contain **0 exact duplicate rows** in each of the three source files.

### Data Type Correction

The `Date` column is converted into datetime format:

```python
campaign_df["Date"] = pd.to_datetime(
    campaign_df["Date"],
    errors="coerce"
)
```

Using `errors="coerce"` converts invalid date values into `NaT` instead of stopping the preprocessing process.

### Other Preprocessing

- Numerical and categorical feature preparation
- Data type correction
- Feature transformation
- Categorical encoding
- Numerical scaling where required by the model

---

# 3. Exploratory Data Analysis (EDA)

EDA is performed to understand campaign behavior before Machine Learning.

### Univariate Analysis

- Revenue distribution
- ROI distribution
- Numerical feature distributions
- Outlier detection

### Bivariate / Multivariate Analysis

- Brand-wise campaign performance
- Campaign comparison
- Spend/Acquisition Cost vs Revenue
- Clicks, Leads and Conversions relationships
- ROI analysis
- Channel performance
- Correlation analysis

### Key EDA Purpose

EDA helps identify patterns, relationships, unusual observations, and performance differences across brands, campaigns, and marketing channels before model development.

---

# 4. Feature Engineering

## Profit_Flag

A binary target is created from ROI:

```text
Profit_Flag = 1 → Profit Campaign
Profit_Flag = 0 → Loss Campaign
```

This converts the profitability problem into a binary classification problem.

## Channel_Used — Multi-Label Encoding

The `Channel_Used` field can contain multiple channels in one value, for example:

```text
Facebook, Instagram
WhatsApp, YouTube
Facebook, Google, YouTube
```

Because a campaign can use more than one channel, **multi-label encoding** is applied so that each relevant marketing channel can be represented as a separate model feature.

Channels represented in the uploaded data include:

- Facebook
- Instagram
- YouTube
- Google
- Email
- WhatsApp

## ROI and Data Leakage

`ROI` is excluded from the classification feature set because it is directly related to campaign profitability.

This reduces the risk of the classification model learning from information that is too closely related to the target.

---

# 5. Model Building

The dataset is divided into training and testing data so that the models can be evaluated on unseen data.

## Regression — Revenue Prediction

**Target:** `Revenue`

Models:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

### Why these models?

- **Linear Regression:** Provides a simple baseline for continuous prediction.
- **Decision Tree Regressor:** Captures non-linear relationships and rule-based patterns.
- **Random Forest Regressor:** Combines multiple decision trees and can capture more complex relationships.

---

## Classification — Profit/Loss Prediction

**Target:** `Profit_Flag`

Models:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

### Why these models?

- **Logistic Regression:** Provides a simple baseline for binary classification.
- **Decision Tree Classifier:** Captures non-linear classification patterns.
- **Random Forest Classifier:** Combines multiple trees and can provide robust classification performance.

---

# 6. Model Evaluation

## Regression Metrics

The regression models are evaluated using:

- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R² Score** — explains the proportion of target variance captured by the model

### Reported Regression Performance

| Model | R² Score |
|---|---:|
| Linear Regression | 0.762 |
| Decision Tree Regressor | 0.826 |
| **Random Forest Regressor** | **0.908** |

### Final Regression Model

**Random Forest Regressor**

**Reported R² = 0.908**

It was selected because it achieved the highest reported R² score among the evaluated regression models.

---

## Classification Metrics

The classification models are evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

### Reported Classification Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 92.54% |
| Decision Tree Classifier | 94.80% |
| **Random Forest Classifier** | **96.00%** |

### Final Classification Model

**Random Forest Classifier**

**Reported Accuracy = 96.00%**

It was selected because it achieved the highest reported accuracy among the evaluated classification models.

---

# 7. Insights & Reporting

The project identifies important factors associated with campaign profitability and performance, including:

- Customer Segment
- Campaign Type
- Marketing Channel
- Engagement Score
- Leads
- Conversions
- Acquisition Cost

These factors are used to support business interpretation and marketing recommendations.

---

# 8. Business Recommendations

Based on the project analysis and model outputs:

1. **Invest more in high-performing campaigns**
2. **Prioritize marketing channels that generate better conversions**
3. **Improve customer targeting and segmentation**
4. **Reduce spending on low-performing campaigns**
5. **Use prediction before making major campaign investments**
6. **Optimize marketing budget allocation**
7. **Monitor campaign performance continuously**

---

# 9. Streamlit Application

A Streamlit application is developed to provide an interactive interface for the trained models.

### Application Features

Users can:

- Enter campaign details
- Provide campaign performance inputs
- Predict expected Revenue
- Predict Profit/Loss
- View prediction-related recommendations
- Understand key prediction inputs

The application should apply the **same preprocessing and feature-engineering logic used during model training** before generating predictions.

---

# 10. Technologies Used

### Programming
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn

### Visualization
- Matplotlib
- Seaborn
- Plotly

### Deployment
- Streamlit

### Model Persistence
- Joblib

---

# 11. Project Structure

A typical project structure is:

```text
Marketing_Campaign_Performance_Prediction/
│
├── nykaa_campaign_data_with_nulls.csv
├── purplle_campaign_data_with_nulls.csv
├── tira_campaign_data_with_nulls.csv
│
├── cleaned_marketing_campaign_data.csv
├── feature_engineered_marketing_campaign_data.csv
│
├── Data_Collection.py
├── Data_Preprocessing.py
├── Feature_Engineering.py
├── Step4_EDA.py
├── Step5_Model_Building.py
├── Step6_Model_Evaluation.py
├── Step7_Insights_Reporting.py
│
├── app.py
│
├── regression_models.pkl
├── classification_models.pkl
│
├── Regression_Model_Evaluation.csv
├── Classification_Model_Evaluation.csv
│
└── README.md
```

> Keep the filenames above only if they match the actual files in your project folder.

---

# 12. Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib
```

---

# 13. Running the Project

### Run the Streamlit application

```bash
streamlit run app.py
```

The application opens in the browser and provides the interactive prediction interface.

---

# 14. Future Enhancements

Possible future improvements include:

- Further hyperparameter tuning and optimization
- Advanced Machine Learning algorithms
- Real-time campaign monitoring
- Automated marketing budget optimization
- Cloud deployment
- Model monitoring and periodic retraining

---

# 15. Project Deliverables

The project deliverables include:

- Raw marketing campaign CSV files
- Cleaned/preprocessed dataset
- Feature-engineered dataset
- Python preprocessing and modeling scripts
- Regression model
- Classification model
- Model evaluation reports
- Streamlit prediction application
- EDA and business insights
- Project documentation

---

# 16. Conclusion

This project demonstrates an end-to-end Machine Learning workflow for **multi-brand marketing campaign analytics**.

The solution transforms raw campaign data into structured features, analyzes campaign performance, predicts **Revenue**, and classifies campaigns as **Profit or Loss**.

The reported evaluation results show that **Random Forest Regressor achieved the best regression performance with an R² of 0.908**, while **Random Forest Classifier achieved the best classification performance with 96% accuracy**.

The resulting insights can help marketing teams improve campaign selection, customer targeting, channel strategy, and budget allocation.

---

## 📊 Project Summary

| Component | Implementation |
|---|---|
| Domain | Marketing Analytics & Business Intelligence |
| Brands | Nykaa, Purplle, Tira |
| Raw Records | 166,665 |
| Raw Features per File | 16 |
| Combined Features | 17 including Brand |
| Regression Target | Revenue |
| Classification Target | Profit_Flag |
| Channel Processing | Multi-label Encoding |
| Missing Values | Median / Most Frequent |
| Regression Models | Linear Regression, Decision Tree, Random Forest |
| Classification Models | Logistic Regression, Decision Tree, Random Forest |
| Best Regression Model | Random Forest Regressor |
| Reported Regression R² | 0.908 |
| Best Classification Model | Random Forest Classifier |
| Reported Classification Accuracy | 96% |
| Deployment | Streamlit |

---

## 👩‍💻 Skills Demonstrated

- Python Programming
- Pandas & NumPy
- Data Cleaning
- Missing Value Handling
- Data Preprocessing
- Multi-Label Encoding
- Feature Engineering
- Exploratory Data Analysis
- Data Visualization
- Regression
- Classification
- Model Evaluation
- Data Leakage Prevention
- Model Comparison
- Streamlit Deployment
- Business Insight Generation
- End-to-End Machine Learning Pipeline
