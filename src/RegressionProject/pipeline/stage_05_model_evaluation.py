from RegressionProject.components.model_evaluation import ModelEvaluation
from RegressionProject.config.configuration import ConfigurationManager

STAGE_NAME = "Model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            dags_hub_config = config.get_dags_hub_config()
            model_evaluation_config = ModelEvaluation(model_evaluation_config,dags_hub_config)
            model_evaluation_config.log_into_mlflow()
        except Exception as e:
            raise e




