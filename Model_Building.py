import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*70)
print("STEP 5 : MODEL BUILDING")
print("="*70)

df = pd.read_csv(os.path.join(BASE_DIR,"feature_engineered_marketing_campaign_data.csv"))

if "Profit_Flag" not in df.columns:
    df["Profit_Flag"] = ((df["Revenue"]-df["Acquisition_Cost"])>0).astype(int)

df.drop(columns=["Campaign_ID","Campaign_Name","Customer_ID","Unnamed: 0"],errors="ignore",inplace=True)
df.drop(columns=["Date"], errors="ignore", inplace=True)

num=df.select_dtypes(include=np.number).columns
cat=df.select_dtypes(include="object").columns

df[num]=df[num].fillna(df[num].median())
for c in cat:
    df[c]=df[c].fillna("Unknown")

df=pd.get_dummies(df,columns=cat,drop_first=True)

X_reg=df.drop(columns=["Revenue","Profit_Flag"],errors="ignore")
y_reg=df["Revenue"]

X_cls=df.drop(columns=["Profit_Flag","Revenue","ROI"],errors="ignore")
y_cls=df["Profit_Flag"]

Xtr_r,Xte_r,ytr_r,yte_r=train_test_split(X_reg,y_reg,test_size=0.2,random_state=42)
Xtr_c,Xte_c,ytr_c,yte_c=train_test_split(X_cls,y_cls,test_size=0.2,random_state=42,stratify=y_cls)

joblib.dump(list(X_reg.columns),os.path.join(MODEL_DIR,"regression_features.pkl"))
joblib.dump(list(X_cls.columns),os.path.join(MODEL_DIR,"classification_features.pkl"))

reg_models={
"Linear Regression":LinearRegression(),
"Decision Tree Regressor":DecisionTreeRegressor(random_state=42),
"Random Forest Regressor":RandomForestRegressor(n_estimators=30,random_state=42,n_jobs=-1)
}

results=[]
for name,model in reg_models.items():
    model.fit(Xtr_r,ytr_r)
    pred=model.predict(Xte_r)
    joblib.dump(model,os.path.join(MODEL_DIR,name.replace(" ","_")+".pkl"))
    results.append([name,
                    mean_squared_error(yte_r,pred),
                    np.sqrt(mean_squared_error(yte_r,pred)),
                    mean_absolute_error(yte_r,pred),
                    r2_score(yte_r,pred)])
print(pd.DataFrame(results,columns=["Model","MSE","RMSE","MAE","R2"]))

grid=GridSearchCV(RandomForestRegressor(random_state=42,n_jobs=-1),
                  {"n_estimators":[30],"max_depth":[10]},
                  cv=3,scoring="r2",n_jobs=-1)
grid.fit(Xtr_r,ytr_r)
best_reg=grid.best_estimator_

cls_models={
"Logistic Regression":LogisticRegression(max_iter=500),
"Decision Tree Classifier":DecisionTreeClassifier(random_state=42),
"Random Forest Classifier":RandomForestClassifier(n_estimators=30,random_state=42,n_jobs=-1)
}
cres=[]
trained={}
for name,model in cls_models.items():
    model.fit(Xtr_c,ytr_c)
    pred=model.predict(Xte_c)
    joblib.dump(model,os.path.join(MODEL_DIR,name.replace(" ","_")+".pkl"))
    trained[name]=model
    cres.append([name,
                 accuracy_score(yte_c,pred),
                 precision_score(yte_c,pred,zero_division=0),
                 recall_score(yte_c,pred,zero_division=0),
                 f1_score(yte_c,pred,zero_division=0)])
cdf=pd.DataFrame(cres,columns=["Model","Accuracy","Precision","Recall","F1"])
print(cdf)
best_cls=trained[cdf.loc[cdf["F1"].idxmax(),"Model"]]

joblib.dump(best_reg,os.path.join(MODEL_DIR,"Best_Regression_Model.pkl"))
joblib.dump(best_cls,os.path.join(MODEL_DIR,"Best_Classification_Model.pkl"))

joblib.dump(Xte_r,os.path.join(MODEL_DIR,"X_test_reg.pkl"))
joblib.dump(yte_r,os.path.join(MODEL_DIR,"y_test_reg.pkl"))
joblib.dump(Xte_c,os.path.join(MODEL_DIR,"X_test_cls.pkl"))
joblib.dump(yte_c,os.path.join(MODEL_DIR,"y_test_cls.pkl"))

print("Completed successfully.")