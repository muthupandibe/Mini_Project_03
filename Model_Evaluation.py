# ==========================================================
# MODEL EVALUATION
# ==========================================================
import os, joblib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
MODEL_DIR=os.path.join(BASE_DIR,"models")

print("="*60)
print("STEP 6 : MODEL EVALUATION")
print("="*60)

# Load test data
Xr=joblib.load(os.path.join(MODEL_DIR,"X_test_reg.pkl"))
yr=joblib.load(os.path.join(MODEL_DIR,"y_test_reg.pkl"))
Xc=joblib.load(os.path.join(MODEL_DIR,"X_test_cls.pkl"))
yc=joblib.load(os.path.join(MODEL_DIR,"y_test_cls.pkl"))

# Regression models
reg_files={
"Linear Regression":"Linear_Regression.pkl",
"Decision Tree Regressor":"Decision_Tree_Regressor.pkl",
"Random Forest Regressor":"Random_Forest_Regressor.pkl"
}

reg_results=[]
best_r2=-1
best_reg=None
for name,file in reg_files.items():
    model=joblib.load(os.path.join(MODEL_DIR,file))
    pred=model.predict(Xr)
    mse=mean_squared_error(yr,pred)
    rmse=np.sqrt(mse)
    mae=mean_absolute_error(yr,pred)
    r2=r2_score(yr,pred)
    reg_results.append([name,mse,rmse,mae,r2])
    if r2>best_r2:
        best_r2=r2; best_reg=name

reg_df=pd.DataFrame(reg_results,columns=["Model","MSE","RMSE","MAE","R2 Score"])
print("\nRegression Comparison")
print(reg_df)
print("\nBest Regression Model:",best_reg)

# Classification models
cls_files={
"Logistic Regression":"Logistic_Regression.pkl",
"Decision Tree Classifier":"Decision_Tree_Classifier.pkl",
"Random Forest Classifier":"Random_Forest_Classifier.pkl"
}
cls_results=[]
best_acc=-1
best_name=None
best_pred=None
for name,file in cls_files.items():
    model=joblib.load(os.path.join(MODEL_DIR,file))
    pred=model.predict(Xc)
    acc=accuracy_score(yc,pred)
    pre=precision_score(yc,pred,zero_division=0)
    rec=recall_score(yc,pred,zero_division=0)
    f1=f1_score(yc,pred,zero_division=0)
    cls_results.append([name,acc,pre,rec,f1])
    if acc>best_acc:
        best_acc=acc; best_name=name; best_pred=pred

cls_df=pd.DataFrame(cls_results,columns=["Model","Accuracy","Precision","Recall","F1 Score"])
print("\nClassification Comparison")
print(cls_df)
print("\nBest Classification Model:",best_name)

reg_df.to_csv(os.path.join(BASE_DIR,"Regression_Model_Comparison.csv"),index=False)
cls_df.to_csv(os.path.join(BASE_DIR,"Classification_Model_Comparison.csv"),index=False)

print("\nClassification Report\n")
print(classification_report(yc,best_pred))
print("Confusion Matrix")
print(confusion_matrix(yc,best_pred))

plt.figure(figsize=(5,5))
plt.scatter(yr, joblib.load(os.path.join(MODEL_DIR,reg_files[best_reg])).predict(Xr))
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.show()

print("\nEvaluation completed.")