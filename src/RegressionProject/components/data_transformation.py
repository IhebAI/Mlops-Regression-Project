from RegressionProject.entity import DataTransformationConfig
from RegressionProject.logging import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from RegressionProject.utils.common import save_object_pkl


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_object(self):
        try:

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logger.info(f"Categorical columns: {self.config.categorical_columns}")
            logger.info(f"Numerical columns: {self.config.numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("Numerical_pipeline", numerical_pipeline, self.config.numerical_columns),
                    ("Categorical_pipeline", categorical_pipeline, self.config.categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise e

    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)
        train, test = train_test_split(data, test_size=0.25, random_state=42)
        train.to_csv(self.config.data_train, index=False)
        test.to_csv(self.config.data_test, index=False)
        logger.info("Split data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

    def load_data(self):
        train_df = pd.read_csv(self.config.data_train)
        test_df = pd.read_csv(self.config.data_test)
        logger.info("Read train and test data is completed")
        return train_df, test_df

    def separate_features_and_target(self, test_df, train_df, target_column_name):
        input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
        target_feature_train_df = train_df[target_column_name]
        logger.info("Separating X and Y for train data is completed")

        input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
        target_feature_test_df = test_df[target_column_name]
        logger.info("Separating X and Y for test data is completed")
        return input_feature_train_df, target_feature_train_df, input_feature_test_df, target_feature_test_df

    def apply_preprocessing(self, preprocessing_obj, input_feature_train_df, input_feature_test_df,
                            target_feature_train_df, target_feature_test_df):
        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
        logger.info(f"Preprocessing object is applied on training and testing data.")
        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
        return train_arr, test_arr

    def save_transformed_data(self, train_arr, test_arr):
        train_transformed_df = pd.DataFrame(train_arr)
        test_transformed_df = pd.DataFrame(test_arr)
        train_transformed_df.to_csv(self.config.transformed_data_train, index=False)
        test_transformed_df.to_csv(self.config.transformed_data_test, index=False)
        logger.info("Saved transformed train and test data as CSV files.")

    def save_preprocessor(self, preprocessor):
        try:
            save_object_pkl(file_path=self.config.preprocessor_obj_file_path, obj=preprocessor)
            logger.info("Saved preprocessing object.")
        except Exception as e:
            logger.error("Failed to save preprocessing object:", e)

    def run_data_processing_pipeline(self):
        try:
            # Get preprocessor
            preprocessor = self.get_data_transformer_object()

            # Split data
            self.train_test_splitting()

            # Load data
            train_df, test_df = self.load_data()

            # Separate features and target
            input_feature_train_df, target_feature_train_df, input_feature_test_df, target_feature_test_df = self.separate_features_and_target(
                test_df, train_df, self.config.target_column)

            # Apply preprocessing
            train_arr, test_arr = self.apply_preprocessing(preprocessor, input_feature_train_df, input_feature_test_df,
                                                           target_feature_train_df, target_feature_test_df)

            # Save transformed data
            self.save_transformed_data(train_arr, test_arr)

            # Save preprocessor
            self.save_preprocessor(preprocessor)

            logger.info("Data processing pipeline completed successfully.")
        except Exception as e:
            logger.error("Error in data processing pipeline:", e)
