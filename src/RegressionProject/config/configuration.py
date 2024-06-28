from RegressionProject.constants import *
from RegressionProject.entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, \
    ModelTrainerConfig, ModelEvaluationConfig
from RegressionProject.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH,
            schema_filepath=SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        schema = self.schema

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            data_train=config.data_train,
            data_test=config.data_test,
            transformed_data_train=config.transformed_data_train,
            transformed_data_test=config.transformed_data_test,
            preprocessor_obj_file_path=config.preprocessor_obj_file_path,
            numerical_columns=config.numerical_columns,
            categorical_columns=config.categorical_columns,
            target_column=schema.TARGET_COLUMN.name
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        data_transformation_config = self.config.data_transformation
        train_config = self.config.model_trainer

        create_directories([train_config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=train_config.root_dir,
            model_name=train_config.model_name,
            trained_model_file_path=train_config.trained_model_file_path,
            grid_search_evaluation_result=train_config.grid_search_evaluation_result,
            transformed_train_data_path=data_transformation_config.transformed_data_train,
            transformed_test_data_path=data_transformation_config.transformed_data_test

        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config_model_eval = self.config.model_evaluation
        config_model_train = self.config.model_trainer
        config_model_transform = self.config.data_transformation

        params = self.params
        schema = self.schema.TARGET_COLUMN

        create_directories([config_model_eval.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config_model_eval.root_dir,
            metric_file_name=config_model_eval.metric_file_name,
            mlflow_uri=config_model_eval.mlflow_uri,
            transformed_data_test=config_model_transform.transformed_data_test,
            transformed_data_train=config_model_transform.transformed_data_train,
            model_path=config_model_train.trained_model_file_path,
            target_column=schema.name,

            all_params=params,

        )
        return model_evaluation_config
