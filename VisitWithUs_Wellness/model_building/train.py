from huggingface_hub import hf_hub_download
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize Hugging Face API
api = HfApi()

# Load dataset from Hugging Face dataset repo
repo_id = "sureshsharma4747/Customer-Purchase-Prediction"

Xtrain_path = hf_hub_download(repo_id=repo_id, filename="Xtrain.csv", repo_type="dataset")
Xtest_path  = hf_hub_download(repo_id=repo_id, filename="Xtest.csv", repo_type="dataset")
ytrain_path = hf_hub_download(repo_id=repo_id, filename="ytrain.csv", repo_type="dataset")
ytest_path  = hf_hub_download(repo_id=repo_id, filename="ytest.csv", repo_type="dataset")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()  # flatten to 1D
ytest = pd.read_csv(ytest_path).values.ravel()

# Features
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender',
    'ProductPitched', 'MaritalStatus', 'Designation'
]

# Handle class imbalance
import numpy as np

classes, counts = np.unique(ytrain, return_counts=True)
class_weight = counts[0] / counts[1]

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Base model
#xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, use_label_encoder=False, eval_metric="logloss")

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow experiment
mlflow.set_experiment("Customer_Purchase_Classification")
with mlflow.start_run():
    # Train with hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Best model
    best_model = grid_search.best_estimator_

    # Log best hyperparameters
    mlflow.log_params(grid_search.best_params_)

    # Evaluation
    classification_threshold = 0.45
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    acc = accuracy_score(ytest, y_pred_test)
    precision = precision_score(ytest, y_pred_test)
    recall = recall_score(ytest, y_pred_test)
    f1 = f1_score(ytest, y_pred_test)
    print("Classification Report (Test Data):")
    print(classification_report(ytest, y_pred_test))

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save and log model
    model_path = "best_customer_purchase_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.sklearn.log_model(best_model, "model")
    print(f"✅ Model saved locally at {model_path}")

    # Upload to Hugging Face Model Hub
    model_repo_id = "sureshsharma4747/Customer-Purchase-Model"
    try:
        api.repo_info(repo_id=model_repo_id, repo_type="model")
        print(f"Model repo '{model_repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{model_repo_id}' not found. Creating new repo...")
        create_repo(repo_id=model_repo_id, repo_type="model", private=False)
        print(f"Model repo '{model_repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=model_repo_id,
        repo_type="model",
    )

    print(f"🚀 Model uploaded successfully: https://huggingface.co/{model_repo_id}")
