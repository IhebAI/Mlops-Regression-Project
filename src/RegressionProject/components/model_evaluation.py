from pathlib import Path
from RegressionProject.entity import ModelEvaluationConfig, DagsHubConfig
from RegressionProject.utils.common import save_json, load_object_pkl, read_transformed_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import dagshub
import mlflow.sklearn
import numpy as np

import subprocess
from RegressionProject.utils.common import load_best_model_from_json


class ModelEvaluation:
    def __init__(self, evaluation_config: ModelEvaluationConfig, dags_hub_config: DagsHubConfig):
        self.evaluation_config = evaluation_config
        self.dags_hub_config = dags_hub_config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def get_git_info(self):
        repo_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).strip().decode()
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
        return repo_url, commit_hash, branch_name

    def log_into_mlflow(self):

        repo_url, commit_hash, branch_name = self.get_git_info()

        dagshub.init(repo_owner=self.dags_hub_config.repo_owner, repo_name=self.dags_hub_config.repo_name,
                     mlflow=self.dags_hub_config.mlflow)

        _, _, test_x, test_y = read_transformed_data(self.evaluation_config.transformed_data_train,
                                                     self.evaluation_config.transformed_data_test)

        # Load the model from the pickle file
        model = load_object_pkl(self.evaluation_config.model_path)

        mlflow.set_registry_uri(self.evaluation_config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        predicted_math_scores = model.predict(test_x)
        rmse, mae, r2 = self.eval_metrics(test_y, predicted_math_scores)

        # Saving metrics as local
        scores = {"rmse": rmse, "mae": mae, "r2": r2}
        save_json(path=Path(self.evaluation_config.metric_file_name), data=scores)

        best_model_name, results = load_best_model_from_json("artifacts/model_trainer/train_results.json")

        with mlflow.start_run():
            mlflow.log_artifact('params.yaml', artifact_path="all_models_params/all_models_and_params")
            mlflow.log_artifact('artifacts/model_trainer/train_results.json',
                                artifact_path="all_models_params/best_models_params")
            mlflow.log_artifact('schema.yaml', artifact_path="schema")
            mlflow.log_artifact('requirements.txt', artifact_path="training_requirements")
            mlflow.log_artifact('config/config.yaml', artifact_path="configuration")
            mlflow.log_artifact('logs/running_logs.log', artifact_path="training_logs")

            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_param("repo_url", repo_url)
            mlflow.log_param("branch_name", branch_name)
            mlflow.log_param("commit_hash", commit_hash)
            for param in results[best_model_name]['best_params']:
                mlflow.log_param(param, results[best_model_name]['best_params'][param])

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="best_regression_model")
            else:
                mlflow.sklearn.log_model(model, "model")
