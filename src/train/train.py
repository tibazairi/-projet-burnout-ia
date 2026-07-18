import argparse
import os
import pandas as pd
import joblib
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# En local, Azure ML ne configure pas MLFLOW_TRACKING_URI -> on force alors
# un backend SQLite pour contourner le bug Windows avec les espaces dans le chemin utilisateur.
# Sur Azure ML, MLFLOW_TRACKING_URI est deja configure automatiquement -> on ne touche a rien.
if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str)
parser.add_argument("--model_output", type=str)

# Hyperparametres exposes en arguments -> le sweep va faire varier ces valeurs
parser.add_argument("--n_estimators", type=int, default=200)
parser.add_argument("--max_depth", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=0.05)
parser.add_argument("--subsample", type=float, default=0.8)

args = parser.parse_args()

mlflow.sklearn.autolog()

train = pd.read_csv(f"{args.train_data}/train.csv")

X = train.drop("Burnout_Risk_Level", axis=1)
y = train["Burnout_Risk_Level"]

params = {
    "n_estimators": args.n_estimators,
    "max_depth": args.max_depth,
    "learning_rate": args.learning_rate,
    "subsample": args.subsample,
    "random_state": 42,
}

with mlflow.start_run():
    mlflow.log_params(params)

    model = GradientBoostingClassifier(**params)
    model.fit(X, y)

    y_pred = model.predict(X)
    train_accuracy = accuracy_score(y, y_pred)

    print("=== Performance sur donnees d'entrainement ===")
    print(classification_report(y, y_pred))
    print(f"train_accuracy={train_accuracy}")

    # Metrique cible pour le sweep -- doit etre loggee explicitement avec mlflow.log_metric
    mlflow.log_metric("train_accuracy", train_accuracy)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=== Top 5 variables les plus predictives du Burnout ===")
    print(importances.head(5))

    for feature, importance in importances.head(5).items():
        mlflow.log_metric(f"importance_{feature}", importance)

joblib.dump(model, f"{args.model_output}/model.pkl")