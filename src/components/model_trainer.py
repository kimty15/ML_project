import os
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomExpection
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "KNN": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                }
            # Evaluate model
            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models=models)

            ## Get the best model score from dict
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]
            threshold = 0.9 
            if best_model_score < threshold: 
                logging.info(f"Can't find model greater than {threshold}")
                logging.info(f"Best found model on test dataset is {best_model_name} with score {best_model_score}")
                return best_model_score
            else:
                logging.info(f"Best found model on test dataset is {best_model_name} ")

                save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                )
                predicted = best_model.predict(X_test)
                r2 = r2_score(y_test, predicted)
                return r2
        
        except Exception as e:
            logging.error("Error occured in model training method")
            raise CustomExpection(e, sys)
