# Model Bulding
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

df = pd.read_csv("feature_engineered_marketing_campaign_data.csv")

TARGET = "Revenue"

X = df.drop(columns=["Revenue", "Profit_Flag"], errors="ignore")
y = df["Revenue"]

X = X.drop(columns=["Campaign_ID", "Campaign_Name", "Customer_ID", "Date", "Channel_Used"], errors="ignore")

categorical_cols = X.select_dtypes(
    include=["object", "string", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]).columns.tolist()

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), 
                                      ("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                          ("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numerical_cols), 
                                               ("cat", categorical_transformer, categorical_cols)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "XGBRegressor": XGBRegressor(random_state=42, objective="reg:squarederror", n_jobs=-1),
    "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_state=42),
    "AdaBoost Regressor": AdaBoostRegressor(random_state=42)
}

hyper_para_config = {
    "Linear Regression": {},
    "Lasso": {"model__alpha": [0.01, 0.1, 1]},
    "Ridge": {"model__alpha": [0.01, 0.1, 1, 10]},
    "K-Neighbors Regressor": {"model__n_neighbors": [3, 5]},
    "Decision Tree": {"model__max_depth": [5, 10]},
    "Random Forest Regressor": {"model__n_estimators": [30], "model__max_depth": [10]},
    "Gradient Boosting Regressor": {"model__n_estimators": [50], "model__learning_rate": [0.1]},
    "XGBRegressor": {"model__n_estimators": [50], "model__max_depth": [3], "model__learning_rate": [0.1]},
    "CatBoosting Regressor": {"model__iterations": [50], "model__depth": [6], "model__learning_rate": [0.1]},
    "AdaBoost Regressor": {"model__n_estimators": [50], "model__learning_rate": [0.1]}
}

def evaluate(name, y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return {"Model": name, "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}

results = []
best_estimators = {}
best_params = {}

for name, model in models.items():
    print("\nTraining:", name)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    grid = GridSearchCV(estimator=pipeline, param_grid=hyper_para_config[name], cv=3, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_estimator = grid.best_estimator_
    best_estimators[name] = best_estimator
    best_params[name] = grid.best_params_
    y_pred = best_estimator.predict(X_test)
    results.append(evaluate(name, y_test, y_pred))
    print("Best Parameters:", grid.best_params_)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2", ascending=False).reset_index(drop=True)

print("\nMODEL EVALUATION RESULTS")
print(results_df.to_string(index=False))

best_model_name = results_df.loc[0, "Model"]
best_model = best_estimators[best_model_name]
best_model_params = best_params[best_model_name]

print("\nBEST MODEL:", best_model_name)
print("BEST PARAMETERS:", best_model_params)
print("BEST R2:", round(results_df.loc[0, "R2"], 4))

results_df.to_csv("Model_Evaluation_Results.csv", index=False)
print("\nModel_Evaluation_Results.csv Saved Successfully")


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================================
# CLASSIFICATION
# ==========================================================

TARGET = "Profit_Flag"

X_cls = df.drop(
    columns=["Profit_Flag", "Revenue", "ROI"],
    errors="ignore"
)

y_cls = df[TARGET]

X_cls = X_cls.drop(
    columns=["Campaign_ID", "Campaign_Name", "Customer_ID", "Date", "Channel_Used"],
    errors="ignore"
)

categorical_cols = X_cls.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numeric_cols = X_cls.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor_cls = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.20,
    random_state=42,
    stratify=y_cls
)

classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "K-Neighbors Classifier": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(random_state=42, n_jobs=-1),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
    "AdaBoost Classifier": AdaBoostClassifier(random_state=42)
}

classification_params = {
    "Logistic Regression": {
        "model__C": [1]
    },
    "K-Neighbors Classifier": {
        "model__n_neighbors": [5]
    },
    "Decision Tree Classifier": {
        "model__max_depth": [10]
    },
    "Random Forest Classifier": {
        "model__n_estimators": [30],
        "model__max_depth": [10]
    },
    "Gradient Boosting Classifier": {
        "model__n_estimators": [50],
        "model__learning_rate": [0.1]
    },
    "AdaBoost Classifier": {
        "model__n_estimators": [50],
        "model__learning_rate": [0.1]
    }
}

def evaluate_classification(name, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

classification_results = []
best_classification_models = {}
best_classification_params = {}

for name, model in classification_models.items():

    print("\nTraining:", name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_cls),
            ("model", model)
        ]
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=classification_params[name],
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(
        X_train_cls,
        y_train_cls
    )

    best_estimator = grid.best_estimator_

    best_classification_models[name] = best_estimator
    best_classification_params[name] = grid.best_params_

    y_pred = best_estimator.predict(
        X_test_cls
    )

    result = evaluate_classification(
        name,
        y_test_cls,
        y_pred
    )

    classification_results.append(result)

    print(
        "Best Parameters:",
        grid.best_params_
    )

classification_results_df = pd.DataFrame(
    classification_results
)

classification_results_df = classification_results_df.sort_values(
    by="F1",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL RESULTS")
print("=" * 70)

print(
    classification_results_df.to_string(
        index=False
    )
)

best_classification_name = classification_results_df.loc[
    0,
    "Model"
]

best_classification_model = best_classification_models[
    best_classification_name
]

best_classification_params = best_classification_params[
    best_classification_name
]

print("\nBEST CLASSIFICATION MODEL:")
print(best_classification_name)

print("\nBEST PARAMETERS:")
print(best_classification_params)

print(
    "\nBEST F1 SCORE:",
    round(
        classification_results_df.loc[0, "F1"],
        4
    )
)

classification_results_df.to_csv(
    "Classification_Model_Evaluation.csv",
    index=False
)

print(
    "\nClassification_Model_Evaluation.csv Saved Successfully"
)  


import os
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/Best_Regression_Model.pkl"
)

joblib.dump(
    best_classification_model,
    "models/Best_Classification_Model.pkl"
)

print("\nBest Regression Model Saved Successfully")
print("Best Classification Model Saved Successfully")

best_model_name = results_df.loc[0, "Model"]
best_model = best_estimators[best_model_name]

best_classification_name = classification_results_df.loc[0, "Model"]
best_classification_model = best_classification_models[best_classification_name]

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/Best_Regression_Model.pkl")
joblib.dump(best_classification_model, "models/Best_Classification_Model.pkl")

joblib.dump(X_test, "models/X_test_reg.pkl")
joblib.dump(y_test, "models/y_test_reg.pkl")

joblib.dump(X_test_cls, "models/X_test_cls.pkl")
joblib.dump(y_test_cls, "models/y_test_cls.pkl")

print("\nBest Regression Model:", best_model_name)
print("Best Classification Model:", best_classification_name)
print("\nAll required model files saved successfully.")
