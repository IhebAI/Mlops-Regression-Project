from pathlib import Path
from RegressionProject.entity import ModelEvaluationConfig
from RegressionProject.utils.common import save_json, load_object_pkl, read_transformed_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import dagshub
import mlflow.sklearn
import numpy as np


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        dagshub.init(repo_owner='iheb.aamrii', repo_name='Mlops-Regression-Project', mlflow=True)

        _, _, test_x, test_y = read_transformed_data(self.config.transformed_data_train, self.config.transformed_data_test)

        # Load the model from the pickle file
        model=load_object_pkl(self.config.model_path)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_math_scores = model.predict(test_x)
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_math_scores)

            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

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
