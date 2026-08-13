# Marketing Campaign Performance Prediction

## 1. Project Overview

**Marketing Campaign Performance Prediction** is an end-to-end Machine Learning project in the **Marketing Analytics** domain.

The project analyzes campaign data from **Nykaa, Purplle, and Tira** to understand campaign performance and build predictive models for:

- **Revenue Prediction** — Regression
- **Profit/Loss Prediction** — Binary Classification

The project includes data collection, preprocessing, EDA, feature engineering, model building, **GridSearchCV hyperparameter tuning**, model evaluation, feature-importance analysis, business insights, and Streamlit deployment.

---

## 2. Business Problem

Marketing campaigns generate data such as impressions, clicks, leads, conversions, acquisition cost, revenue, ROI, customer segment, campaign type, and marketing channels.

The business needs a data-driven solution to estimate expected campaign revenue and identify whether a campaign is likely to be profitable or loss-making.

---

## 3. Objectives

1. Combine campaign data from Nykaa, Purplle, and Tira.
2. Clean and preprocess the raw campaign data.
3. Handle missing values and duplicate records.
4. Correct data types, including the `Date` column.
5. Perform Exploratory Data Analysis (EDA).
6. Transform the multi-valued `Channel_Used` feature.
7. Create the binary `Profit_Flag` target.
8. Build regression models for Revenue prediction.
9. Build classification models for Profit/Loss prediction.
10. Apply **GridSearchCV** for hyperparameter tuning.
11. Compare base and tuned model performance.
12. Evaluate the final models using appropriate metrics.
13. Generate feature-importance and business insights.
14. Deploy the prediction system using Streamlit.

---

# 4. Project Workflow

```text
Raw CSV Files
      ↓
Data Collection
      ↓
Data Cleaning & Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Base Model Building
      ↓
GridSearchCV Hyperparameter Tuning
      ↓
Model Evaluation & Comparison
      ↓
Feature Importance & Insights
      ↓
Streamlit Deployment
```

---

# 5. Dataset

The project uses three raw campaign datasets:

- `nykaa_campaign_data_with_nulls.csv`
- `purplle_campaign_data_with_nulls.csv`
- `tira_campaign_data_with_nulls.csv`

The three datasets are combined into:

- `combined_marketing_campaign_data.csv`

A `Brand` column is used to identify the source brand.

### Main feature groups

**Campaign Features**
- Campaign_ID
- Campaign_Type
- Date
- Duration

**Customer Features**
- Target_Audience
- Customer_Segment
- Language

**Performance Features**
- Impressions
- Clicks
- Leads
- Conversions
- Engagement_Score

**Financial Features**
- Acquisition_Cost
- Revenue
- ROI

**Channel Feature**
- Channel_Used

**Engineered Features**
- Brand
- Profit_Flag
- Multi-label channel features

---

# 6. Data Collection

The three brand-specific CSV files are loaded using Pandas.

A `Brand` column is added to preserve the identity of each source dataset, and the three datasets are combined into a single dataset for unified analysis and Machine Learning.

---

# 7. Data Cleaning & Preprocessing

## Missing Values

- Numerical columns → **Median imputation**
- Categorical columns → **Most-frequent-value (mode) imputation**

Median is used for numerical variables because it is less sensitive to extreme values, while mode is appropriate for categorical variables.


## Other preprocessing

- Data type correction
- Numerical feature preparation
- Categorical feature preparation
- Feature transformation
- Scaling where required by the algorithm

---

# 8. Exploratory Data Analysis (EDA)

EDA is performed before Machine Learning to understand the structure and behavior of the campaign data.

### Univariate Analysis

- Revenue distribution
- ROI distribution
- Numerical feature distributions
- Outlier detection

### Bivariate Analysis

- Acquisition Cost vs Revenue
- Leads vs Conversions
- Campaign performance comparisons
- Brand-wise performance

### Multivariate Analysis

- Correlation heatmap
- Channel and campaign performance
- Relationships among performance variables
- Revenue and ROI patterns across campaign dimensions

### EDA Outputs

The project generates analytical outputs such as:

- `EDA_Summary.csv`
- `Revenue_Correlation.csv`
- `Top_10_Campaigns.csv`
- `Bottom_10_Campaigns.csv`
- `brand_revenue.html`

These outputs support the interpretation of campaign performance.

---

# 9. Feature Engineering

## Profit_Flag

A binary target is created for classification:

```text
1 → Profit Campaign
0 → Loss Campaign
```

### Regression Target

```text
Revenue
```

### Classification Target

```text
Profit_Flag
```

---

## Channel_Used — Multi-Label Encoding

A campaign can use more than one marketing channel.

For example:

```text
Facebook, Instagram
Facebook, Google, YouTube
WhatsApp, Email
```

Therefore, `Channel_Used` is treated as a **multi-label feature** rather than a normal single categorical variable.

The individual channels are transformed into separate model features.

A `channel_classes.pkl` file is created to preserve the channel-class information required during prediction.

---

## Data Leakage Prevention

`ROI` is removed from the classification feature set because it is directly related to profitability.

This helps prevent the classification model from using information that is too closely related to the target and producing unrealistically strong results.

---

# 10. Model Building

The project contains two Machine Learning tasks.

## A. Regression — Revenue Prediction

**Target:** `Revenue`

### Algorithms

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

### Why these algorithms?

**Linear Regression**  
Used as a simple baseline to model a continuous target.

**Decision Tree Regressor**  
Used because it can capture non-linear relationships and decision rules.

**Random Forest Regressor**  
Used because it combines multiple decision trees and can capture more complex patterns.

---

## B. Classification — Profit/Loss Prediction

**Target:** `Profit_Flag`

### Algorithms

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

### Why these algorithms?

**Logistic Regression**  
Used as a simple baseline for binary classification.

**Decision Tree Classifier**  
Used to capture non-linear classification patterns.

**Random Forest Classifier**  
Used as an ensemble model capable of learning complex relationships through multiple decision trees.

---

# 11. Hyperparameter Tuning — GridSearchCV

**GridSearchCV was used in this project for hyperparameter tuning.**

The purpose of GridSearchCV is to systematically evaluate combinations of model hyperparameters using cross-validation and identify a strong parameter configuration.

### Why GridSearchCV?

Instead of relying only on default model parameters, GridSearchCV helps identify better hyperparameter combinations based on cross-validated model performance.

The workflow is:

```text
Base Model
    ↓
Define Parameter Grid
    ↓
GridSearchCV
    ↓
Cross-Validation
    ↓
Best Parameters
    ↓
Best/Tuned Model
    ↓
Test Set Evaluation
```

The project therefore includes both **base model comparison** and **tuned model selection**.

> The exact parameter grid should be kept consistent with the `Model_Building.py` code used in the project.

---

# 12. Model Evaluation

## Regression Metrics

The regression models are evaluated using:

- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R² Score**

### Reported Regression Results

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

### Reported Classification Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 92.54% |
| Decision Tree Classifier | 94.80% |
| **Random Forest Classifier** | **96.00%** |

### Final Classification Model

**Random Forest Classifier**

**Reported Accuracy = 96.00%**

It was selected because it achieved the highest reported classification accuracy among the evaluated models.

---

# 13. Model Comparison & Evaluation Files

The project generates comparison and evaluation outputs, including:

- `Regression_Comparison.csv`
- `Regression_Model_Comparison.csv`
- `Classification_Comparison.csv`
- `Classification_Model_Comparison.csv`
- `Regression_Feature_Importance.csv`
- `Feature_Importance.csv`

These files are used to compare model performance and analyze important predictive features.

---

# 14. Feature Importance

Feature-importance analysis is used to understand which input variables contribute most to model predictions.

The project contains:

```text
Feature_Importance.csv
Regression_Feature_Importance.csv
```

These outputs support interpretation of the Machine Learning model and help translate technical results into business recommendations.

---

# 15. Business Insights

The project focuses on factors such as:

- Customer Segment
- Campaign Type
- Marketing Channel
- Engagement Score
- Leads
- Conversions
- Acquisition Cost

These factors can help explain campaign performance and support marketing decisions.

---

# 16. Business Recommendations

Based on the analysis and predictions:

1. Invest more in high-performing campaigns.
2. Focus on channels that generate better engagement and conversions.
3. Improve customer targeting and segmentation.
4. Reduce spending on consistently low-performing campaigns.
5. Use predicted revenue before making major campaign investments.
6. Use profitability predictions to support campaign selection.
7. Optimize marketing budget allocation.
8. Monitor campaign performance continuously.

---

# 17. Streamlit Application

The project includes:

```text
app.py
```

The Streamlit application provides an interactive interface for using the trained Machine Learning models.

### Main purpose

- Accept campaign-related input
- Apply the required preprocessing
- Predict expected Revenue
- Predict Profit/Loss
- Present prediction results and recommendations

The saved preprocessing/model information is reused during prediction so that the application follows the same feature structure used during model training.

---

# 18. Project Structure

Based on the project files, the main structure is:

```text
PROJECT_NO_03/
│
├── models/
│
├── app.py
│
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
│
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
│
├── Regression_Feature_Importance.csv
├── Feature_Importance.csv
│
└── Top_10_Campaigns.csv
```

The `models/` folder contains the saved Machine Learning model artifacts used by the application.

---

# 19. Technologies Used

### Programming
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn

### Model Tuning
- GridSearchCV
- Cross-validation

### Visualization
- Matplotlib
- Seaborn
- Plotly

### Deployment
- Streamlit

### Model Persistence
- Joblib / Pickle-based model artifacts

---

# 20. Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scriptsctivate
```

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib
```

---

# 21. Running the Project

Run the Streamlit application from the project directory:

```bash
streamlit run app.py
```

The application can then be opened in the browser.

---

# 22. Future Enhancements

Possible improvements include:

- More extensive hyperparameter optimization
- Advanced Machine Learning algorithms
- Real-time campaign monitoring
- Automated marketing budget optimization
- Cloud deployment
- Model monitoring and periodic retraining
- Automated model retraining using new campaign data

---

# 23. Conclusion

This project demonstrates an end-to-end Machine Learning solution for **marketing campaign performance prediction**.

The project combines multi-brand campaign data, performs preprocessing and EDA, engineers predictive features, builds regression and classification models, and applies **GridSearchCV hyperparameter tuning** to improve model selection.

The reported results identify **Random Forest Regressor** as the strongest regression model with an R² score of **0.908**, and **Random Forest Classifier** as the strongest classification model with **96% reported accuracy**.

The project also produces feature-importance and comparison reports and provides a Streamlit application for interactive prediction.

---

## 📊 Project Summary

| Component | Details |
|---|---|
| Domain | Marketing Analytics |
| Brands | Nykaa, Purplle, Tira |
| Regression Target | Revenue |
| Classification Target | Profit_Flag |
| Channel Processing | Multi-Label Encoding |
| Missing Values | Median / Mode |
| EDA | Univariate, Bivariate, Multivariate |
| Regression Models | Linear Regression, Decision Tree, Random Forest |
| Classification Models | Logistic Regression, Decision Tree, Random Forest |
| Hyperparameter Tuning | **GridSearchCV** |
| Regression Evaluation | MSE, RMSE, MAE, R² |
| Classification Evaluation | Accuracy, Precision, Recall, F1 |
| Best Reported Regression Model | Random Forest Regressor |
| Best Reported R² | **0.908** |
| Best Reported Classification Model | Random Forest Classifier |
| Best Reported Accuracy | **96%** |
| Deployment | Streamlit |

---

## 👩‍💻 Skills Demonstrated

- Python
- Pandas
- NumPy
- Data Cleaning
- Missing Value Imputation
- Date-Time Processing
- Exploratory Data Analysis
- Multi-Label Encoding
- Feature Engineering
- Regression
- Classification
- GridSearchCV
- Cross-Validation
- Model Evaluation
- Feature Importance
- Data Leakage Prevention
- Model Comparison
- Streamlit Deployment
- Business Insight Generation
