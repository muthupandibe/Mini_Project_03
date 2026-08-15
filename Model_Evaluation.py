# MODEL EVALUATION
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# TARGET THRESHOLDS
R2_TARGET = 0.95
ACCURACY_TARGET = 0.95

# LOAD BEST MODELS

best_regression_model = joblib.load(
    "models/Best_Regression_Model.pkl"
)

best_classification_model = joblib.load(
    "models/Best_Classification_Model.pkl"
)

print("\nBest models loaded successfully.")

# LOAD TEST DATA 
X_test_reg = joblib.load(
    "models/X_test_reg.pkl"
)

y_test_reg = joblib.load(
    "models/y_test_reg.pkl"
)

X_test_cls = joblib.load(
    "models/X_test_cls.pkl"
)

y_test_cls = joblib.load(
    "models/y_test_cls.pkl"
)

print("\nRegression Test Data Shape:", X_test_reg.shape)
print("Classification Test Data Shape:", X_test_cls.shape)

# REGRESSION MODEL EVALUATION

print("REGRESSION MODEL EVALUATION")

y_pred_reg = best_regression_model.predict(
    X_test_reg
)

r2 = r2_score(
    y_test_reg,
    y_pred_reg
)

mae = mean_absolute_error(
    y_test_reg,
    y_pred_reg
)

mse = mean_squared_error(
    y_test_reg,
    y_pred_reg
)

rmse = np.sqrt(mse)

mape = mean_absolute_percentage_error(
    y_test_reg,
    y_pred_reg
)

print("\nRegression Metrics")
print("R²   :", round(r2, 4))
print("MAE  :", round(mae, 4))
print("MSE  :", round(mse, 4))
print("RMSE :", round(rmse, 4))
print("MAPE :", round(mape * 100, 2), "%")

# REGRESSION TARGET CHECK

if r2 >= R2_TARGET:
    regression_target_status = "Achieved"
else:
    regression_target_status = "Not Achieved"

print("\nRegression Target")
print("Required R² :", R2_TARGET)
print("Actual R²   :", round(r2, 4))
print("Status      :", regression_target_status)

# ACTUAL VS PREDICTED REVENUE
plt.figure(figsize=(8, 5))

plt.scatter(
    y_test_reg,
    y_pred_reg,
    alpha=0.5
)

plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")

plt.tight_layout()
plt.show()

# PERFECT PREDICTION REFERENCE LINE
plt.figure(figsize=(8, 5))

plt.scatter(
    y_test_reg,
    y_pred_reg,
    alpha=0.5
)

minimum_value = min(
    y_test_reg.min(),
    y_pred_reg.min()
)

maximum_value = max(
    y_test_reg.max(),
    y_pred_reg.max()
)

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    linestyle="--"
)

plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue with Reference Line")

plt.tight_layout()
plt.show()

# CLASSIFICATION MODEL EVALUATION

y_pred_cls = best_classification_model.predict(
    X_test_cls
)

accuracy = accuracy_score(
    y_test_cls,
    y_pred_cls
)

precision = precision_score(
    y_test_cls,
    y_pred_cls,
    zero_division=0
)

recall = recall_score(
    y_test_cls,
    y_pred_cls,
    zero_division=0
)

f1 = f1_score(
    y_test_cls,
    y_pred_cls,
    zero_division=0
)

print("\nClassification Metrics")
print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1 Score  :", round(f1, 4))

# CLASSIFICATION TARGET CHECK

if accuracy >= ACCURACY_TARGET:
    classification_target_status = "Achieved"
else:
    classification_target_status = "Not Achieved"

print("\nClassification Target")
print("Required Accuracy :", ACCURACY_TARGET)
print("Actual Accuracy   :", round(accuracy, 4))
print("Status            :", classification_target_status)

# CLASSIFICATION REPORT

print("\nClassification Report")

print(
    classification_report(
        y_test_cls,
        y_pred_cls,
        zero_division=0
    )
)

# CONFUSION MATRIX

cm = confusion_matrix(
    y_test_cls,
    y_pred_cls
)

print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()

# ROC-AUC

roc_auc = None

if hasattr(
    best_classification_model,
    "predict_proba"
):

    y_probability = (
        best_classification_model
        .predict_proba(X_test_cls)[:, 1]
    )

    roc_auc = roc_auc_score(
        y_test_cls,
        y_probability
    )

    fpr, tpr, thresholds = roc_curve(
        y_test_cls,
        y_probability
    )

    print("\nROC-AUC :", round(roc_auc, 4))

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()
    plt.show()

else:

    print(
        "\nROC-AUC cannot be calculated because "
        "the model does not support predict_proba."
    )

# SAVE REGRESSION RESULTS

regression_results = pd.DataFrame([
    {
        "Model": "Best Regression Model",
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2_Target": R2_TARGET,
        "Target_Status": regression_target_status
    }
])

regression_results.to_csv(
    "Regression_Evaluation.csv",
    index=False
)

# SAVE CLASSIFICATION RESULTS

classification_results = pd.DataFrame([
    {
        "Model": "Best Classification Model",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "Accuracy_Target": ACCURACY_TARGET,
        "Target_Status": classification_target_status
    }
])

# FINAL MODEL EVALUATION SUMMARY

print("FINAL MODEL EVALUATION SUMMARY")

print("\nREGRESSION")
print("-" * 40)
print("R²   :", round(r2, 4))
print("MAE  :", round(mae, 2))
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("MAPE :", round(mape * 100, 2), "%")
print("Target R² :", R2_TARGET)
print("Target Status :", regression_target_status)

print("\nCLASSIFICATION")
print("-" * 40)
print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1 Score  :", round(f1, 4))

if roc_auc is not None:
    print("ROC-AUC   :", round(roc_auc, 4))

print("Target Accuracy :", ACCURACY_TARGET)
print("Target Status   :", classification_target_status)

# FINAL PROJECT TARGET SUMMARY

print("PROJECT PERFORMANCE TARGET SUMMARY")

print(
    "\nRegression Target : R² >= 0.95"
)

if r2 >= R2_TARGET:
    print(
        "Regression Result: TARGET ACHIEVED"
    )
else:
    print(
        "Regression Result: TARGET NOT ACHIEVED"
    )

print(
    "\nClassification Target : Accuracy >= 0.95"
)

if accuracy >= ACCURACY_TARGET:
    print(
        "Classification Result: TARGET ACHIEVED"
    )
else:
    print(
        "Classification Result: TARGET NOT ACHIEVED"
    )
