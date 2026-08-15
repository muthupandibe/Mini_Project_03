# 📊 Multi-Brand Marketing Campaign Performance Prediction

## 📌 Project Overview

The **Multi-Brand Marketing Campaign Performance Prediction** project uses Data Science and Machine Learning techniques to analyze marketing campaign data and predict campaign performance.

The project covers:

- Data Collection
- Data Preprocessing
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Regression
- Classification
- Feature Selection
- GridSearchCV Hyperparameter Tuning
- Model Evaluation
- Business Insights
- Streamlit Application

The system predicts:

1. **Campaign Revenue** using Regression
2. **Campaign Profit/Loss** using Classification

---

## 🎯 Project Objectives

- Analyze marketing campaign performance across multiple brands.
- Clean and preprocess campaign data.
- Handle missing values and duplicate records.
- Create meaningful engineered features.
- Analyze campaign performance using EDA.
- Predict campaign revenue.
- Predict campaign profitability.
- Compare multiple Machine Learning algorithms.
- Tune model parameters using GridSearchCV.
- Evaluate models using appropriate metrics.
- Build an interactive Streamlit application.
- Generate data-driven business recommendations.

---

## 📊 Dataset

The final combined dataset contains:

- **161,249 records**
- **17 original columns**

Important columns include:

| Column | Description |
|---|---|
| Campaign_ID | Unique campaign identifier |
| Campaign_Type | Type of marketing campaign |
| Target_Audience | Target customer group |
| Duration | Campaign duration |
| Channel_Used | Marketing channel |
| Impressions | Number of impressions |
| Clicks | Number of clicks |
| Leads | Number of leads |
| Conversions | Number of conversions |
| Revenue | Revenue generated |
| Acquisition_Cost | Customer acquisition cost |
| ROI | Return on Investment |
| Language | Campaign language |
| Engagement_Score | Customer engagement score |
| Customer_Segment | Customer segment |
| Date | Campaign date |
| Brand | Brand associated with campaign |

---

# 📂 Project Structure

```text
Marketing_Campaign_Performance_Prediction/
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
├── EDA_Summary.csv
├── brand_revenue.html
├── Regression_Evaluation.csv
├── Classification_Evaluation.csv
├── channel_classes.pkl
│
├── models/
│   ├── Best_Regression_Model.pkl
│   ├── Best_Classification_Model.pkl
│   ├── X_test_reg.pkl
│   ├── y_test_reg.pkl
│   ├── X_test_cls.pkl
│   └── y_test_cls.pkl
│
└── README.md
```

---

# 🛠️ Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Joblib
- Streamlit

### Development Tools
- VS Code
- Jupyter Notebook / Python
- Command Prompt

---

# 1️⃣ Data Collection

Marketing campaign datasets from multiple brands are collected and combined.

During data collection:

- Individual datasets are loaded.
- Brand information is added.
- Datasets are combined.
- The combined dataset is saved.

### Output

```text
combined_marketing_campaign_data.csv
```

---

# 2️⃣ Data Preprocessing

The preprocessing stage prepares the raw dataset for analysis and Machine Learning.

### Main Steps

#### Duplicate Removal

Duplicate campaign records are removed using `Campaign_ID` when available.

#### Date Conversion

The `Date` column is converted to datetime format using:

```python
pd.to_datetime(
    campaign_df["Date"],
    dayfirst=True,
    errors="coerce"
)
```

Invalid dates are converted to missing values.

#### Numerical Cleaning

Numerical columns are converted using `pd.to_numeric()`.

Infinite values are replaced with missing values.

Missing numerical values are filled using the median.

#### Categorical Cleaning

Categorical columns are:

- Converted to string
- Stripped of unnecessary spaces
- Missing values handled using mode
- Text standardized

#### Negative Value Handling

Negative values are removed from business-related positive fields such as:

- Duration
- Impressions
- Clicks
- Leads
- Conversions
- Revenue
- Acquisition Cost
- Engagement Score

#### Business Rules

The following relationships are validated:

```text
Clicks <= Impressions
Leads <= Clicks
Conversions <= Leads
```

#### ROI Validation

ROI values are checked for unrealistic values and extreme values are replaced using the median ROI.

### Output

```text
cleaned_marketing_campaign_data.csv
```

---

# 3️⃣ Feature Engineering

Feature Engineering creates additional variables that help the Machine Learning models understand campaign performance.

## Engineered Features

### CTR

```text
CTR = Clicks / Impressions
```

### Conversion Rate

```text
Conversion_Rate = Conversions / Clicks
```

### Cost Per Click

```text
Cost_Per_Click = Acquisition_Cost / Clicks
```

### Cost Per Conversion

```text
Cost_Per_Conversion = Acquisition_Cost / Conversions
```

### Lead Conversion Rate

```text
Lead_Conversion_Rate = Conversions / Leads
```

Zero denominators are safely handled to prevent division-by-zero errors.

---

## 💰 Profit Flag

A binary `Profit_Flag` is created based on ROI:

```text
ROI > 0  → Profit_Flag = 1
ROI <= 0 → Profit_Flag = 0
```

Therefore:

```text
1 = Profit
0 = Loss
```

---

## 📣 Channel Encoding

The `Channel_Used` column is transformed using:

```python
MultiLabelBinarizer
```

This creates channel features such as:

```text
Channel_Email
Channel_Facebook
Channel_Google
Channel_Instagram
Channel_Whatsapp
Channel_Youtube
```

The channel classes are saved as:

```text
channel_classes.pkl
```

### Output

```text
feature_engineered_marketing_campaign_data.csv
```

---

# 4️⃣ Exploratory Data Analysis (EDA)

EDA is performed to understand campaign performance and relationships between variables.

### Analysis Performed

- Campaign count by brand
- Revenue by brand
- Average ROI by brand
- Campaign type distribution
- Target audience distribution
- Revenue distribution
- ROI distribution
- Revenue outlier analysis
- Correlation heatmap
- Engagement Score vs Revenue
- CTR vs Revenue
- Profit vs Loss distribution
- Top 10 revenue campaigns
- Marketing channel analysis

### EDA Outputs

```text
EDA_Summary.csv
brand_revenue.html
```

---

# 5️⃣ Model Building

Two Machine Learning problems are developed.

## 💰 Regression

### Target

```text
Revenue
```

### Algorithms

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

---

## 🎯 Classification

### Target

```text
Profit_Flag
```

### Algorithms

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## 🔀 Train-Test Split

The dataset is divided into:

```text
Training Data
Testing Data
```

The test data remains unseen during model training and is used for final validation.

---

## ⚠️ Data Leakage Prevention

`Profit_Flag` is created from ROI.

Therefore, ROI is **excluded from classification features** to prevent target leakage.

This ensures that the classification model does not directly receive the information used to construct its target.

---

## 🔧 Hyperparameter Tuning

`GridSearchCV` is used to tune model parameters and identify better model configurations.

The best models are saved using Joblib:

```text
models/Best_Regression_Model.pkl
models/Best_Classification_Model.pkl
```

---

# 6️⃣ Model Evaluation

The trained models are evaluated using unseen test data.

## Regression Metrics

- R² Score
- MAE
- MSE
- RMSE

Project target:

```text
R² >= 0.95
```

### Metric Interpretation

**R²:** Higher is better.

**MAE:** Lower is better.

**MSE:** Lower is better.

**RMSE:** Lower is better.

---

## Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

Project target:

```text
Accuracy >= 0.95
```

### Metric Interpretation

**Accuracy:** Percentage of correct predictions.

**Precision:** Correctness of positive predictions.

**Recall:** Ability to identify actual profitable campaigns.

**F1 Score:** Balance between Precision and Recall.

**ROC-AUC:** Ability to distinguish profitable and loss-making campaigns.

---

# 🏆 Best Model Selection

The best regression model is selected based on:

- Higher R²
- Lower RMSE
- Lower MAE
- Lower MSE

The best classification model is selected based on:

- Higher Accuracy
- Higher Precision
- Higher Recall
- Higher F1 Score

---

# 7️⃣ Insights & Reporting

The project generates business-oriented insights from the campaign data and model results.

The analysis helps identify:

- High-performing brands
- Low-performing campaigns
- High-revenue campaigns
- Profitable campaigns
- Effective marketing channels
- Important campaign performance factors
- Campaigns requiring optimization

---

# 💡 Business Recommendations

The system can support marketing decisions by recommending actions such as:

### For profitable campaigns

- Continue the campaign strategy.
- Consider increasing the marketing budget.
- Focus on successful target audiences.
- Scale successful campaigns.
- Continue monitoring ROI and conversion rate.

### For loss-making campaigns

- Review the campaign strategy.
- Reduce acquisition cost.
- Improve CTR.
- Improve conversion rate.
- Test alternative marketing channels.
- Re-evaluate the target audience.

---

# 8️⃣ Streamlit Application

An interactive Streamlit application is developed for campaign analysis and prediction.

## Application Pages

### 🏠 Dashboard

Displays:

- Total campaigns
- Total revenue
- Average revenue
- Average ROI
- Brand-wise revenue
- Campaign type performance
- Target audience analysis
- Channel analysis

### 🔮 Prediction

Users can enter:

- Campaign Type
- Target Audience
- Brand
- Language
- Customer Segment
- Marketing Channel
- Duration
- Impressions
- Clicks
- Leads
- Conversions
- Acquisition Cost
- Engagement Score

The application calculates:

- CTR
- Conversion Rate
- Cost Per Click
- Cost Per Conversion
- Lead Conversion Rate

### Prediction Outputs

The application displays:

- Predicted Revenue
- Profit/Loss
- Prediction Confidence
- Campaign performance chart
- Marketing funnel
- Campaign summary
- Business recommendation

A prediction report can also be downloaded as CSV.

---

# ▶️ How to Run the Project

## Step 1: Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib
```

## Step 2: Run Data Collection

```bash
python Data_Collection.py
```

## Step 3: Run Data Preprocessing

```bash
python Data_Preprocessing.py
```

## Step 4: Run Feature Engineering

```bash
python Feature_Engineering.py
```

## Step 5: Run EDA

```bash
python Step4_EDA.py
```

## Step 6: Run Model Building

```bash
python Step5_Model_Building.py
```

## Step 7: Run Model Evaluation

```bash
python Step6_Model_Evaluation.py
```

## Step 8: Run Insights & Reporting

```bash
python Step7_Insights_Reporting.py
```

## Step 9: Launch Streamlit Application

```bash
streamlit run app.py
```

---

# 🔄 Complete Project Workflow

```text
Raw Marketing Data
        ↓
Data Collection
        ↓
Data Preprocessing
        ↓
Cleaned Dataset
        ↓
Feature Engineering
        ↓
Feature-Engineered Dataset
        ↓
Exploratory Data Analysis
        ↓
Feature Selection
        ↓
Train/Test Split
        ↓
Model Training
        ↓
GridSearchCV
        ↓
Best Model Selection
        ↓
Model Evaluation
        ↓
Insights & Reporting
        ↓
Streamlit Application
        ↓
Revenue + Profit/Loss Prediction
        ↓
Business Recommendation
```

---

# 📁 Important Output Files

| File | Purpose |
|---|---|
| `combined_marketing_campaign_data.csv` | Combined raw dataset |
| `cleaned_marketing_campaign_data.csv` | Cleaned dataset |
| `feature_engineered_marketing_campaign_data.csv` | Feature-engineered dataset |
| `channel_classes.pkl` | Saved channel encoding classes |
| `EDA_Summary.csv` | EDA summary |
| `brand_revenue.html` | Interactive brand revenue chart |
| `Regression_Evaluation.csv` | Regression evaluation results |
| `Classification_Evaluation.csv` | Classification evaluation results |
| `Best_Regression_Model.pkl` | Best regression model |
| `Best_Classification_Model.pkl` | Best classification model |

---

# 💼 Business Value

This project provides a Machine Learning based decision-support system for marketing teams.

It can help businesses:

- Estimate campaign revenue.
- Identify potentially profitable campaigns.
- Compare brand performance.
- Compare marketing channels.
- Understand customer segments.
- Optimize campaign spending.
- Improve campaign conversion performance.
- Support data-driven marketing decisions.

---

# 🧪 Model Validation

Models are evaluated using unseen test data to measure their ability to generalize to new campaign data.

The classification model excludes ROI because ROI is used to create `Profit_Flag`.

This reduces the risk of target leakage.

---

# 🏁 Conclusion

The **Multi-Brand Marketing Campaign Performance Prediction** project demonstrates an end-to-end Data Science and Machine Learning workflow.

```text
Data Collection
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
EDA
      ↓
Machine Learning
      ↓
Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
Business Insights
      ↓
Streamlit Deployment
```

The final application provides interactive campaign analytics and predicts:

- **Campaign Revenue**
- **Campaign Profit/Loss**

The system combines Machine Learning predictions, data visualization, and business recommendations to support better marketing decisions.

---

# 👩‍💻 Skills Demonstrated

- Python
- NumPy
- Pandas
- Data Cleaning
- Data Preprocessing
- Feature Engineering
- Exploratory Data Analysis
- Data Visualization
- Regression
- Classification
- Feature Selection
- Data Leakage Prevention
- GridSearchCV
- Model Evaluation
- Joblib
- Streamlit
- Interactive Dashboard Development
- Business Analytics
- Machine Learning Deployment

---

# ⭐ Project Title

**Multi-Brand Marketing Campaign Performance Prediction Using Machine Learning**

### Project Type

**Data Science / Machine Learning Mini Project**

### Main Predictions

```text
1. Revenue Prediction
2. Profit/Loss Prediction
```

### Application

**Interactive Streamlit Marketing Analytics & Prediction Dashboard**

---

## Author

**Muthupandi**

**Data Science Mini Project**
