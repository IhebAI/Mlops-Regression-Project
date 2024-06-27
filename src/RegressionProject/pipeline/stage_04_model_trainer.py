from RegressionProject.components.model_trainer import ModelTrainer
from RegressionProject.config.configuration import ConfigurationManager


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_params = model_trainer.models_trainer(config.params, model_trainer_config.transformed_train_data_path,
                                                        model_trainer_config.transformed_test_data_path)

        except Exception as e:
            raise e



