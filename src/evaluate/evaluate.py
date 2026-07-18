import argparse
import os
import pandas as pd
import joblib
import json
import mlflow
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# En local, Azure ML ne configure pas MLFLOW_TRACKING_URI -> on force alors
# un backend SQLite pour contourner le bug Windows avec les espaces dans le chemin utilisateur.
# Sur Azure ML, MLFLOW_TRACKING_URI est deja configure automatiquement -> on ne touche a rien.
if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

parser = argparse.ArgumentParser()
parser.add_argument("--test_data", type=str)
parser.add_argument("--model_input", type=str)
parser.add_argument("--metrics_output", type=str)
args = parser.parse_args()

test = pd.read_csv(f"{args.test_data}/test.csv")
model = joblib.load(f"{args.model_input}/model.pkl")

X_test = test.drop("Burnout_Risk_Level", axis=1)
y_test = test["Burnout_Risk_Level"]

with mlflow.start_run():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"=== Precision finale sur donnees de test : {accuracy:.2%} ===")
    report = classification_report(
        y_test, y_pred, target_names=["Low", "Medium", "High"], output_dict=True
    )
    print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))
    print("Matrice de confusion (lignes=reel, colonnes=predit) :")
    print(confusion_matrix(y_test, y_pred))

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision_low", report["Low"]["precision"])
    mlflow.log_metric("test_recall_low", report["Low"]["recall"])
    mlflow.log_metric("test_precision_medium", report["Medium"]["precision"])
    mlflow.log_metric("test_recall_medium", report["Medium"]["recall"])
    mlflow.log_metric("test_precision_high", report["High"]["precision"])
    mlflow.log_metric("test_recall_high", report["High"]["recall"])

with open(f"{args.metrics_output}/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)