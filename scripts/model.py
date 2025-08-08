
"""

The module includes the model features to explore and predict models like XGBoost

"""

import joblib
import xgboost
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import shap

from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy import stats

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

np.random.seed(1889)


class Model:
    
    """
    
    The constructor contains the global variables and creates the finalised the dataframe
    
    Some variables are placed to make it easier to code the logic (e.g. min max sclaing)
    
    Hyperparameters:
        - data -> the data in dataframe object
        - features -> gather all dataframe that are not target
        - target -> gather the target dataframe vector
    
    
    """
    def __init__(self, data, target):
        self.data = data
        self.features = data.drop([target], axis=1)
        self.target = data[[target]]
        
        self.model = None
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        
        
    """"""
    def set_model(self, model):
        self.model = model
    
    
    """
    
    Returns the features vector dataframe
    
    """
    def get_features(self) -> pd.DataFrame:
        return self.features
    
    
    """
    
    Returns the target vector dataframe
    
    """
    def get_target(self) -> pd.DataFrame:
        return self.target
    
    
    """
    
    Trains a model alongside their configurations
    
    Hyperparameters:
        - model -> XGBoost model that can have different configurations

    Azhan comments:
        - The train test split defaintely impact the outcome as it is shuffled by default
        - Since there are several attempts of model, I added it as a hyperparameter
        - The only downside, it's not flexible so there would need to be code edits
    
    """
    def train_model(self, model) -> bool:
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=1889)
        
            model.fit(X_test, y_test, eval_set=[(X_train, y_train)], verbose=10)
            
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            
            self.model = model
        except Exception as error:
            print("There was an error ", error)
            return False
        
        return True
    
    
    
    """
    
    Produces the model evaluations of several scores including:
        - Accuracy scores
        - Confusion Matrix
        - ROC scores
        - Cohen Kappa scores
        - Log scores
        - F1 scores
        - Precision scores
        - Recall scores
        
    Returns true if sucessfully produced. Otherwise, returns false
    
    Hyperparameters:
        - name -> the name to title the statistics
    
    Azhan comments:
        - I could split this out even further into separate functions but it may be overwhelming considering it's just called from module
        - I'm assuming these are all the benchmarks needed?
    
    """
    def model_evaluation(self, name) -> bool:
        
        if (len(self.data) == 0):
            print("Error, the data is empty")
            return False
            
        try:
            X_train_predict = self.model.predict(self.X_train)
            X_test_predict = self.model.predict(self.X_test)
            
            X_train_probability_predict = self.model.predict_proba(self.X_train)[:, 1]
            X_test_probability_predict = self.model.predict_proba(self.X_test)[:, 1]
            
            train_accuracy_score = accuracy_score(self.y_train, X_train_predict)
            test_accuracy_score = accuracy_score(self.y_test, X_test_predict)
            train_confusion_matrix = confusion_matrix(self.y_train, X_train_predict)
            test_confusion_matrix = confusion_matrix(self.y_test, X_test_predict)
            train_roc_auc_score = roc_auc_score(self.y_train, X_train_predict)
            test_roc_auc_score = roc_auc_score(self.y_test, X_test_predict)
            train_cohen_score = self.cohen_kappa_score(self.y_train, X_train_predict)
            test_cohen_score = self.cohen_kappa_score(self.y_test, X_test_predict)
            train_log_score = log_loss(self.y_train, X_train_probability_predict)
            test_log_score = log_loss(self.y_test, X_test_probability_predict)
            test_f1_score = f1_score(self.y_test, X_test_predict)
            test_precision_score = precision_score(self.y_test, X_test_predict)
            test_recall_score = recall_score(self.y_test, X_test_predict)
            
            print(f"The Cohen Kappa score on the training data is {train_cohen_score}")
            print(f"The Cohen Kappa score on the test data is {test_cohen_score}")
            
            print(f"The accuracy on train dataset is {train_accuracy_score}")
            print(f"The accuracy on test dataset is {test_accuracy_score}")
            
            print(f"The train confusion matrix is {train_confusion_matrix}")
            print(f"The test confusion matrix is {test_confusion_matrix}")
            
            print(f"ROC on the train data is {train_roc_auc_score}")
            print(f"ROC on the test data is {test_roc_auc_score}")
            
            print(f"The train log loss is {train_log_score}")
            print(f"The test log loss is {test_log_score}")
            
            print(f"The F1 score is {test_f1_score}")
            print(f"The Precision score is {test_precision_score}")
            print(f"The Recall score is {test_recall_score}")
            
            print("Displays ROC graph")
            self.display_roc_curves(self.y_test, X_test_predict, name)
            
            print("Displays model's performance")
            self.display_model_evaluation(name)
            
        except Exception as error:
            print("Error on model evaluation ", error)
            return False
        
        
        return True
    
    
    """
    
    Displays the ROC graph 
    
    Hyperparameters:
        - y_test -> the test dataframe vector
        - y_pred -> the prediction dataframe vector
    
    Returns true if sucessfully produced. Otherwise, returns false

    Azhan comments:
        - the labels needs changing if another model is required (add another hyperparameter)

    
    """
    def display_roc_curves(self, y_test, y_pred, name) -> bool:
        
        try:
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            random_fpr, random_tpr, _ = roc_curve(y_test, [0 for _ in range(len(y_test))])
            plt.plot(fpr, tpr, marker=".", label="XGBoost")
            plt.plot(random_fpr, random_tpr, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Curve")
            plt.savefig(self.data_path() + f"{name} roc curves.pdf")
            plt.show()
        except Exception as error:
            print("There was an error ", error)
            return False
        
        
        return True
    
    
    
    """
    
    Displays the model's performance specifically for XGBoost (tree and importance diagrams)
    
    Returns true if sucessfully produced. Otherwise, returns false


    Azhan comments:
        - Perhaps separating them could be an improvements to avoid coupling
    
    
    """
    def display_model_evaluation(self, name) -> bool:
    
        try:
            
            # importance diagram
            figure_importance, ax_importance = plt.subplots(figsize=(12, 6))
            xgboost.plot_importance(self.model, ax=ax_importance)
            plt.savefig(self.data_path() + f"{name} model importance.pdf")
            plt.show()

            # tree diagram
            figure_tree, ax_tree = plt.subplots(figsize=(16, 16))
            xgboost.plot_tree(self.model, rankdir="LR", ax=ax_tree)
            plt.savefig(self.data_path() + f"{name} model tree.pdf")
            plt.show()            
        except Exception as error:
            print("There was an error ", error)
            return False
        
        
        return True
        
    
    """
    
    
    Calculates the Cohen Kappa score
    
    Hyperparameters:
        - y_test -> the test dataframe vector
        - y_pred -> the prediction dataframe vector
    
    Returns score value

    Azhan comments:
        - I'm assuming quadratic is the only weights he's interested?
    
    
    """
    def cohen_kappa_score(self, y_test, y_pred) -> int:
        y1 = y_test.astype("int").values.tolist()
        y2 = np.clip(np.round(np.array(y_pred)), np.min(y1), np.max(y1)).astype("int")
        
        return round(cohen_kappa_score(y2, y1, weights="quadratic"), 2)
        
        
    """
    
    Performs the cross validation if required
    
    Returns the best parameters to be applied
    
    Azhan comments:
        - Assuming if they want to do more testing, they can do.
        - I'm guessing the range limits are the best ones
    
    """
    def cross_validation(self, model, cv=5, iterations=100) -> {}:
        
        parameter_distributions = {
            "n_estimators": stats.randint(50, 500),
            "learning_rate": stats.uniform(0.01, 0.75),
            "subsample": stats.uniform(0.25, 0.75),
            "max_depth": stats.randint(1, 8),
            "colsample_bytree": stats.uniform(0.1, 0.75),
            "min_child_weight": [1, 3, 5, 7, 9],
        }
        
        parameter_grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameter_distributions,
            cv=cv,
            n_iter=iterations,
            verbose=False,
            scoring="roc_auc",
        )
        
        parameter_grid_search.fit(self.X_train, self.y_train, eval_set=[(self.X_train, self.y_train)], verbose=False)
        
        print("Best parameters are: ", parameter_grid_search.best_params_)
        
        return parameter_grid_search
    
    
    
    """
    
    Display the summary plots
    
    Hyperparameters:
        - enable_categorical -> true if data is caegorical, otherwise, false
        
    Returns true if sucessfully produced. Otherwise, returns false

    Azhan comments:
        - This is restricted to xgboost, maybe not useful for other modelling
    
    """
    def display_summary(self, name, enable_categorical=True) -> bool:
        
        try:
            shap.initjs()

            shap_values = self.model.get_booster().predict(
                xgboost.DMatrix(self.X_train, self.y_train, enable_categorical=enable_categorical), pred_contribs=True
            )
            
            shap.summary_plot(shap_values[:, :-1], self.X_train, show=False)
            
            plt.savefig(self.data_path() + f"{name} summary.pdf")
            plt.show()
        except Exception as error:
            print("There was an error ", error)
            return False
        
        return True
    
    
    
    """
    
    Returns the data file path
    
    """
    def data_path(self) -> str:
        return str(Path(__file__).parent.parent) + "/data/"
    
    
    
    """
    
    Saves the model in a JSON file
    
    Returns true if sucessfully produced. Otherwise, returns false
    
    Azhan comments:
        - I personally prefer pickle or joblib file type
    
    """
    def save_model(self, name):
        try:
            self.model.save_model(self.data_path() + f"{name}.json")
        except Exception as error:
            print("There was an error ", error)
            return False
        
        return True
        
        
        