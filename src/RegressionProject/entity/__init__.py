from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    data_train: Path
    data_test: Path
    transformed_data_train: Path
    transformed_data_test: Path
    preprocessor_obj_file_path: Path
    numerical_columns: list
    categorical_columns: list
    target_column: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_name: str
    trained_model_file_path: Path
    grid_search_evaluation_result: Path
    transformed_train_data_path: Path
    transformed_test_data_path: Path


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    metric_file_name: Path
    mlflow_uri: str
    transformed_data_test: Path
    transformed_data_train: Path
    model_path: Path
    all_params: dict
    target_column: str
