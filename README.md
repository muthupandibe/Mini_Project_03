# Marketing Campaign Performance Prediction

## Overview

Marketing Campaign Performance Prediction is a Machine Learning project that analyzes digital marketing campaign data and predicts campaign revenue and profitability. The project combines data preprocessing, feature engineering, exploratory data analysis (EDA), machine learning model development, and deployment through a Streamlit web application.

## Objectives

- Analyze marketing campaign performance.
- Predict campaign revenue using regression models.
- Classify campaigns as Profit or Loss.
- Compare multiple machine learning algorithms.
- Deploy an interactive Streamlit application for real-time predictions.

## Dataset

The dataset contains campaign-related information such as:

- Campaign Duration
- Impressions
- Clicks
- Leads
- Conversions
- Acquisition Cost
- Campaign Type
- Target Audience
- Channel Used
- Customer Segment
- Language
- Revenue
- ROI

## Project Workflow

### 1. Data Collection
- Combined campaign datasets from multiple brands.
- Loaded data using Pandas.

### 2. Data Preprocessing
- Removed duplicates.
- Handled missing values.
- Treated categorical and numerical features.
- Encoded categorical variables.
- Standardized data where required.

### 3. Feature Engineering
- Created Profit_Flag.
- Created CTR (Click Through Rate).
- Generated additional useful features.

### 4. Exploratory Data Analysis (EDA)
- Revenue Distribution
- ROI Analysis
- Correlation Heatmap
- Campaign-wise Performance
- Channel Performance
- Target Audience Analysis

### 5. Machine Learning Models

#### Regression Models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

**Target:** Revenue

**Evaluation Metrics:**
- RMSE
- MAE
- MSE
- R² Score

#### Classification Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

**Target:** Profit_Flag

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score

### 6. Model Selection

The best-performing regression and classification models are saved using Joblib and used for prediction in the Streamlit application.

### 7. Streamlit Application

The application allows users to:

- Enter campaign details
- Predict expected revenue
- Predict Profit/Loss
- View campaign insights
- Display interactive visualizations

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Joblib
- Streamlit

## Project Structure

```text
Marketing-Campaign-Performance-Prediction/
│
├── data/
│   ├── combined_marketing_campaign_data.csv
│   ├── cleaned_marketing_campaign_data.csv
│   └── feature_engineered_marketing_campaign_data.csv
│
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Building.ipynb
│   └── 05_Model_Evaluation.ipynb
│
├── models/
│   ├── Best_Regression_Model.pkl
│   ├── Best_Classification_Model.pkl
│   ├── regression_models.pkl
│   └── classification_models.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── images/
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Navigate to the project directory:

```bash
cd Marketing-Campaign-Performance-Prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Sample Input

- Campaign Duration
- Impressions
- Clicks
- Leads
- Conversions
- Acquisition Cost
- Campaign Type
- Target Audience
- Channel Used
- Customer Segment
- Language

## Output

- Predicted Revenue
- Profit/Loss Prediction
- Interactive Charts
- Campaign Insights

## Future Enhancements

- Hyperparameter tuning
- Deep Learning models
- Real-time API integration
- Cloud deployment
- Automated report generation

## Results

The project successfully predicts campaign revenue and campaign profitability using supervised machine learning techniques. The Streamlit dashboard enables users to make real-time predictions and visualize campaign performance through an easy-to-use interface.

## Author

**Muthupandi**

**Skills:** Data Science | Machine Learning | Python | SQL | Streamlit
