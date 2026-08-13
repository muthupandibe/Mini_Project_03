# Marketing Campaign Performance Prediction

## Project Overview

Marketing Campaign Performance Prediction is a Machine Learning project that analyzes historical marketing campaign data and predicts campaign outcomes.

The project performs:

- Revenue Prediction using Regression Models
- Profit/Loss Prediction using Classification Models
- Campaign Performance Analysis
- Feature Importance Analysis
- Business Insight Generation
- ML Model Deployment using Streamlit


---

# Business Objective

The objective of this project is to help marketing teams:

- Understand campaign performance
- Predict expected revenue
- Identify profitable and loss-making campaigns
- Discover important factors affecting profitability
- Improve marketing strategies using data-driven decisions


---

# Dataset Description

The dataset contains marketing campaign details from multiple brands.

## Features

### Campaign Features

- Campaign_ID
- Brand
- Campaign_Type
- Date
- Duration


### Customer Features

- Target_Audience
- Customer_Segment
- Language


### Marketing Performance Features

- Impressions
- Clicks
- Leads
- Conversions
- Engagement_Score


### Cost and Revenue Features

- Acquisition_Cost
- ROI
- Revenue


### Marketing Channel Features

- Facebook
- Instagram
- Youtube
- Google
- Email
- Whatsapp


---

# Project Workflow

Data Collection
|
Data Cleaning
|
Exploratory Data Analysis
|
Feature Engineering
|
Model Building
|
Model Evaluation
|
Insights & Reporting
|
Streamlit Deployment



---

# Data Preprocessing

Performed preprocessing steps:

- Missing value handling
- Duplicate checking
- Data type correction
  - Date column converted to datetime using `pd.to_datetime()`
  - Invalid date values handled with `errors="coerce"`
- Feature transformation
- Categorical encoding
- Numerical scaling


### Missing Value Treatment

Numerical Features:

- Median Imputation


Categorical Features:

- Most Frequent Value Imputation


---

# Exploratory Data Analysis (EDA)

EDA was performed to understand campaign behavior.

## Analysis Performed

- Revenue distribution analysis
- ROI analysis
- Brand-wise performance analysis
- Campaign comparison
- Correlation analysis
- Outlier detection


## Visualizations

- Revenue Distribution Plot
- ROI Distribution Plot
- Correlation Heatmap
- Top Performing Campaigns
- Channel Performance Analysis


---

# Feature Engineering

Feature engineering was performed to improve model performance.


## Profit Flag Creation

Created a classification target:

Profit_Flag

1 → Profit Campaign

0 → Loss Campaign

The target is a binary classification variable.


## Encoding

Categorical variables were transformed using:

- One Hot Encoding


## Feature Preparation

Separate datasets were created:

### Regression

Target: Revenue


### Classification

Target: Profit_Flag


ROI was removed from classification features to prevent data leakage.


---

# Machine Learning Model Building

## Regression Models

Objective:

Predict campaign revenue.


Algorithms Used:

1. Linear Regression

2. Decision Tree Regressor

3. Random Forest Regressor



## Classification Models

Objective:

Predict campaign profitability.


Algorithms Used:

1. Logistic Regression

2. Decision Tree Classifier

3. Random Forest Classifier



---

# Model Evaluation

## Regression Evaluation Metrics

Used metrics:

- MSE
- RMSE
- MAE
- R² Score


## Regression Performance


| Model | R² Score |
|---|---:|
| Linear Regression | 0.762 |
| Decision Tree Regressor | 0.826 |
| Random Forest Regressor | 0.908 |


### Best Regression Model
Random Forest Regressor

R² Score = 0.908

Selected because it achieved the highest R² score among the evaluated regression models.



---

## Classification Evaluation Metrics

Used metrics:

- Accuracy
- Precision
- Recall
- F1 Score


## Classification Performance


| Model | Accuracy |
|---|---:|
| Logistic Regression | 92.54% |
| Decision Tree Classifier | 94.80% |
| Random Forest Classifier | 96.00% |


### Best Classification Model
Random Forest Classifier

Accuracy = 96%

Selected because it achieved the highest accuracy among the evaluated classification models.


---

# Insights & Reporting

The project identifies important factors affecting campaign profitability.


## Key Factors

- Customer Segment
- Campaign Type
- Marketing Channel
- Engagement Score
- Leads
- Conversions
- Acquisition Cost


---

# Business Recommendations

Based on model insights:

- Invest more in high-performing campaigns
- Select marketing channels with better conversions
- Improve customer targeting
- Reduce low-performing campaign spending
- Predict campaign success before investment
- Optimize marketing budget allocation


---

# Streamlit Application

A Streamlit web application was developed for interactive prediction.


## Application Features

Users can:

- Enter campaign details
- Predict expected revenue
- Predict Profit/Loss
- View campaign recommendations

---

# Project Structure
Marketing_Campaign_Performance_Prediction
|
feature_engineered_marketing_campaign_data.csv
|
Model_Building.py
|
Model_Evaluation.py
|
Insights_Reporting.py
|
app.py
|
regression_models.pkl
|
classification_models.pkl
|
Regression_Model_Evaluation.csv
|
Classification_Model_Evaluation.csv
|
README.md

---

# Technologies Used

## Programming Language

- Python


## Machine Learning

- Scikit-learn


## Data Processing

- Pandas
- NumPy


## Visualization

- Matplotlib
- Seaborn
- Plotly


## Deployment

- Streamlit


## Model Saving

- Joblib


---

# Future Enhancements

- Further hyperparameter tuning and optimization
- Advanced ML algorithms
- Real-time campaign monitoring
- Automated budget optimization
- Cloud deployment


---

---

# Project Scope Notes

- **ELT / ETL:** No separate ELT or ETL pipeline is included in the documented workflow; the project focuses on data collection, preprocessing, EDA, feature engineering, model building, evaluation, reporting, and Streamlit deployment.
- **Data Migration:** No MongoDB-to-SQL data migration is part of this project.
- **Statistical Significance Testing:** No separate hypothesis/significance test is documented; model performance is evaluated using the regression and classification metrics listed above.
- **Class Imbalance:** No separate class-imbalance technique is documented in this README.

# Conclusion

This project demonstrates the application of Machine Learning in marketing analytics.

The developed solution predicts campaign revenue, identifies profitable campaigns, and provides actionable insights to support data-driven marketing decisions.

The Random Forest model achieved the best performance for both revenue prediction and profitability classification.
