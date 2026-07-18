"""
pipeline.py
-----------
Definit les etapes du pipeline Azure ML SDK v2 :
  1. prep_step        -> nettoyage + encodage + split train/test
  2. train_sweep_step -> entrainement avec recherche automatique d'hyperparametres
  3. evaluate_step     -> evaluation sur les donnees de test
"""

from azure.ai.ml import command, Input, Output
from azure.ai.ml.sweep import Choice, Uniform
from azure.ai.ml.dsl import pipeline

ENV = "azureml://registries/azureml/environments/sklearn-1.5/labels/latest"

# ---------------------------------------------------------------------------
# Etape 1 : Preparation des donnees
# ---------------------------------------------------------------------------
prep_step = command(
    name="prep_data",
    display_name="Preparation des donnees",
    code="../src/prep",
    command="python prep.py --input_data ${{inputs.input_data}} --train_output ${{outputs.train_output}} --test_output ${{outputs.test_output}}",
    inputs={"input_data": Input(type="uri_file")},
    outputs={"train_output": Output(type="uri_folder"), "test_output": Output(type="uri_folder")},
    environment=ENV,
)

# ---------------------------------------------------------------------------
# Etape 2 : Definition du composant d'entrainement de base (reutilisable)
# ---------------------------------------------------------------------------
train_component = command(
    name="train_model",
    display_name="Entrainement",
    code="../src/train",
    command=(
        "python train.py "
        "--train_data ${{inputs.train_data}} "
        "--model_output ${{outputs.model_output}} "
        "--n_estimators ${{inputs.n_estimators}} "
        "--max_depth ${{inputs.max_depth}} "
        "--learning_rate ${{inputs.learning_rate}} "
        "--subsample ${{inputs.subsample}}"
    ),
    inputs={
        "train_data": Input(type="uri_folder"),
        "n_estimators": Input(type="integer"),
        "max_depth": Input(type="integer"),
        "learning_rate": Input(type="number"),
        "subsample": Input(type="number"),
    },
    outputs={"model_output": Output(type="uri_folder")},
    environment=ENV,
)

# ---------------------------------------------------------------------------
# Etape 3 : Evaluation du modele
# ---------------------------------------------------------------------------
evaluate_step = command(
    name="evaluate_model",
    display_name="Evaluation",
    code="../src/evaluate",
    command="python evaluate.py --test_data ${{inputs.test_data}} --model_input ${{inputs.model_input}} --metrics_output ${{outputs.metrics_output}}",
    inputs={
        "test_data": Input(type="uri_folder"),
        "model_input": Input(type="uri_folder"),
    },
    outputs={"metrics_output": Output(type="uri_folder")},
    environment=ENV,
)


@pipeline(name="pipeline_burnout_prediction_with_sweep")
def burnout_pipeline(input_data):
    prep = prep_step(input_data=input_data)

    # Construction du sweep DANS la fonction pipeline, avec le vrai train_data
    # passe directement a l'appel (pas de reassignation apres coup)
    train_job = train_component(
        train_data=prep.outputs.train_output,
        n_estimators=Choice([100, 200, 300]),
        max_depth=Choice([3, 4, 6, 8]),
        learning_rate=Uniform(min_value=0.01, max_value=0.2),
        subsample=Uniform(min_value=0.6, max_value=1.0),
    )

    sweep_job = train_job.sweep(
        primary_metric="train_accuracy",
        goal="maximize",
        sampling_algorithm="random",
    )
    sweep_job.set_limits(max_total_trials=8, max_concurrent_trials=1, timeout=3600)

    evaluate = evaluate_step(
        test_data=prep.outputs.test_output,
        model_input=sweep_job.outputs.model_output,
    )

    return {
        "model": sweep_job.outputs.model_output,
        "metrics": evaluate.outputs.metrics_output,
    }