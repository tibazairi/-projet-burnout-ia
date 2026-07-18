"""
run_pipeline.py
----------------
Ce script :
  1. Se connecte a ton workspace Azure ML
  2. Enregistre (ou met a jour) le dataset comme Data Asset
  3. Importe le pipeline defini dans pipeline.py
  4. Le lance sur un compute cluster Azure
"""

import traceback
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

from pipeline import burnout_pipeline

print("DEBUT DU SCRIPT", flush=True)

SUBSCRIPTION_ID = "3c3b6d94-78ba-4830-b73c-a8e2b2835dae"
RESOURCE_GROUP_NAME = "zairitiba24-rg"
WORKSPACE_NAME = "mlops-smartovate"
COMPUTE_NAME = "cpu-cluster"

DATA_PATH = "C:/Users/zairi dhia el hak/Documents/projet_burnout_ia/data/impact_ia_students.csv"
DATA_ASSET_NAME = "ai_students_impact_data"

try:
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP_NAME,
        workspace_name=WORKSPACE_NAME,
    )
    print(f"Connecte au workspace : {WORKSPACE_NAME}", flush=True)

    data_asset = Data(
        path=DATA_PATH,
        type=AssetTypes.URI_FILE,
        description="Dataset impact IA sur etudiants (burnout)",
        name=DATA_ASSET_NAME,
    )
    registered_data = ml_client.data.create_or_update(data_asset)
    print(f"Dataset enregistre : {registered_data.name}, version {registered_data.version}", flush=True)

    pipeline_job = burnout_pipeline(
        input_data=Input(
            type=AssetTypes.URI_FILE,
            path=f"azureml:{registered_data.name}:{registered_data.version}",
        )
    )

    returned_job = ml_client.jobs.create_or_update(
        pipeline_job,
        compute=COMPUTE_NAME,
        experiment_name="burnout_prediction_experiment",
    )

    print("\nPipeline lance avec succes !", flush=True)
    print(f"Nom du job : {returned_job.name}", flush=True)
    print(f"Suivre l'execution ici : {returned_job.studio_url}", flush=True)

except Exception as e:
    print("\n=== ERREUR DETECTEE ===", flush=True)
    print(f"Type: {type(e).__name__}", flush=True)
    print(f"Message: {e}", flush=True)
    print("\n=== TRACEBACK COMPLET ===", flush=True)
    traceback.print_exc()