from datetime import datetime
from typing import Dict, Union

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
import numpy as np
import pandas as pd

from seal.utils.utils import get_hash


class Run(object):
    """
    Run is a core object of SEAL. The main philosophy around this \
    object is to allow model search, hyperparameters tuning as well as \
    provide metrics calculation according to the experiment plan locked at \
    the build moment. 
    
    Once an experiment is built, you can start a run via the \
    :meth:`~seal.experiment.Experiment.start_run()` method by passing a \
    parameter dictionnary.
    This dictionnary should contain the configuration parameters for shaping \
    autosklearn searchspace. For more details, please see \
    https://automl.github.io/auto-sklearn/master/api.html
    
    A Run instance provides 3 principal attributes :
        - model : Fitted model with hyperparameter tuning
        - metrics : A dictionnary containing train and test metrics value
        - run_id : Hash code corresponding to the id of the run instance
    
    Parameters
    ----------
    kwargs: dict, 
    
        problem: str,
            problem name, either "classification" or "regression"
        data_train: dict,
            dictionnary containing X and y in accordance with the splitting \
            strategy defined in :meth:`~seal.experiment.Experiment.build()`
        data_test: dict, 
            dictionnary containing X_test and y_test in accordance with the \
            splitting strategy defined in :meth:`~seal.experiment.Experiment.build()`
        hp_rules: dict,
            configuration parameters for autosklearn hyperoptimisation search \
            space
        metrics: :class:`~seal.metrics.Metrics`, 
            object containing user defined metrics. Metrics are specified at \
            the building step of an :class:`~seal.experiment.Experiment`
    """

    def __init__(self, **kwargs) -> None:

        self.__date = datetime.now()
        self.__run_id = get_hash(date=self.__date)
        
        if kwargs["problem"] == "classification":
            self.__optimiser = AutoSklearnClassifier(
                **kwargs["hp_rules"]
            )
        elif kwargs["problem"] == "regression":
            self.__optimiser = AutoSklearnRegressor(
                **kwargs["hp_rules"]
            )
   
        self.__fit(**kwargs["data_train"])
        
        self.__metrics = {
            "train": kwargs["metrics"](
                **self.__get_predictions(**kwargs["data_train"])
                ), 
            "test": kwargs["metrics"](
                **self.__get_predictions(**kwargs["data_test"])
                )
        }

    def __fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Start a hyper-optimization cycle with autosklearn engine

        Parameters
        ----------
        X: pd.DataFrame (n_samples, n_features)
           Input samples. DataFrame that will be used during the Run's\
           lifecycle.
        y: pd.Series of shape (n_samples, 1)
           The target variable (corresponding to X)

        """
        self.__optimiser.fit(X, y)    
        self.__optimiser.refit(X, y)

    def __get_predictions(self, 
                      X: pd.DataFrame, 
                      y: pd.Series) -> Dict[str, Union[pd.Series, np.ndarray]]:
        """
        Computes predictions in the form of predicted values and probabilities.\
        Returns a dictionnary containing all the necessary arguments for metric\
        calculation (y_true, y_pred, y_score).
        
        For more details about metrics calculation, please see \
        :class:`~seal.metrics.Metrics`

        Parameters
        ----------
        X: pd.DataFrame (n_samples, n_features)
           Input samples. DataFrame that will be used during the Run's lifecycle.
        y: pd.Series of shape (n_samples, 1)
           The target variable (corresponding to X)

        Returns
        -------
        metrics: Dict[str, Union[pd.Series, np.ndarray]]
            Dictionary containing the target variable and predictions 
        """
                
        if isinstance(self.model, AutoSklearnRegressor):
            y_pred = self.model.predict(X)
            y_score = None
        else :
            y_score = self.model.predict_proba(X)
            y_pred = np.argmax(y_score, axis=1)
            
        target_and_predictions = {
            "y_true": y,
            "y_pred": y_pred,
            "y_score": y_score
        }
        return target_and_predictions

    @property
    def metrics(self) -> Dict[str, Dict[str, Union[float, int]]]:
        return self.__metrics

    @property
    def model(self) -> Union[AutoSklearnClassifier, AutoSklearnRegressor]:
        return self.__optimiser
    
    @property
    def run_id(self) -> str:
        return self.__run_id