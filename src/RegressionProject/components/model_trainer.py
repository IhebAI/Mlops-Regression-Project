from pathlib import Path

from RegressionProject.entity import ModelTrainerConfig
from RegressionProject.logging import logger
from RegressionProject.utils.common import load_json, save_json, save_object_pkl

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import pandas as pd


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def prepare_params_models_gs(self, params):
        model_params = {
            'Decision Tree': (DecisionTreeRegressor(), params.Decision_Tree.to_dict()),
            'Random Forest': (RandomForestRegressor(), params.Random_Forest.to_dict()),
            'Gradient Boosting': (GradientBoostingRegressor(), params.Gradient_Boosting.to_dict()),
            'Linear Regression': (LinearRegression(), {}),
            'XGBRegressor': (XGBRegressor(), params.XGBRegressor.to_dict()),
            'CatBoosting Regressor': (CatBoostRegressor(verbose=False), params.CatBoosting_Regressor.to_dict()),
            'AdaBoost Regressor': (AdaBoostRegressor(), params.AdaBoost_Regressor.to_dict()),
        }
        return model_params

    def perform_grid_search(self, model, param_grid, X, y):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X, y)
        logger.info(f"Best params for {model.__class__.__name__}: {grid_search.best_params_}")
        logger.info(f"Best score for {model.__class__.__name__}: {grid_search.best_score_}")
        return grid_search

    def evaluate_models(self, x_train, y_train, x_test, y_test, model_params):
        results = {}
        for model_name, (model, param_grid) in model_params.items():
            logger.info(f"Performing grid search for {model_name}")
            grid_search = self.perform_grid_search(model, param_grid, x_train, y_train)

            best_params = grid_search.best_params_
            model.set_params(**best_params)
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Calculate training and test metrics
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            results[model_name] = {
                'best_params': best_params,
                'model': model,
                'train_model_score': train_model_score,
                'test_model_score': test_model_score
            }
            # Export results to JSON
        save_json(Path(self.config.grid_search_evaluation_result), results)
        logger.info(f"Training results after grid search is exported to {self.config.grid_search_evaluation_result} ")

    def read_transformed_data(self, transformed_train_data_path, transformed_test_data_path):
        train_data = pd.read_csv(transformed_train_data_path)
        test_data = pd.read_csv(transformed_test_data_path)
        target_column = train_data.columns[-1]
        train_x = train_data.drop([target_column], axis=1)
        test_x = test_data.drop([target_column], axis=1)
        train_y = train_data[target_column].values
        test_y = test_data[target_column].values
        return train_x, train_y, test_x, test_y

    def models_trainer(self, params, transformed_train_data_path, transformed_test_data_path):
        train_x, train_y, test_x, test_y = self.read_transformed_data(transformed_train_data_path,
                                                                      transformed_test_data_path)
        model_params = self.prepare_params_models_gs(params)
        self.evaluate_models(train_x, train_y, test_x, test_y, model_params)
        results = load_json(Path(self.config.grid_search_evaluation_result))
        best_model_name = max(results, key=lambda x: results[x]['test_model_score'])
        best_model = results[best_model_name]['model']
        logger.info("Best Model found is : {}".format(best_model))
        save_object_pkl(Path(self.config.trained_model_file_path), best_model, )
