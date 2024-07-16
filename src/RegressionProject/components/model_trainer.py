from RegressionProject.entity import ModelTrainerConfig
from RegressionProject.utils.common import save_json
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from RegressionProject.logging import logger
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from RegressionProject.constants import *
from RegressionProject.utils.common import load_best_model_from_json, save_object_pkl, read_transformed_data
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


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

            # Calculate training metrics
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
            logger.info(f"Training results exported to {self.config.grid_search_evaluation_result} ")
        return results

    def models_trainer(self, params, transformed_train_data_path, transformed_test_data_path):
        train_x, train_y, test_x, test_y = read_transformed_data(transformed_train_data_path,
                                                                 transformed_test_data_path)
        model_params = self.prepare_params_models_gs(params)
        results = self.evaluate_models(train_x, train_y, test_x, test_y, model_params)
        best_model_name, _ = load_best_model_from_json(self.config.grid_search_evaluation_result)
        best_model = results[best_model_name]['model']
        best_model_score = results[best_model_name]['test_model_score']
        logger.info("Best Model found is : {} with a test score {}".format(best_model_name, best_model_score))
        save_object_pkl(self.config.trained_model_file_path, best_model, )
