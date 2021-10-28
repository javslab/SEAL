from collections import namedtuple

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from seal.experiment import Experiment

Data = namedtuple("data", ['X_train','X_test', 'y_train', 'y_test'])

@pytest.fixture(scope='function')
def classification_experiment() -> Experiment:
    return Experiment("test", "classification")

@pytest.fixture(scope='module')
def classification_data():
    X, y = make_classification()
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.5, 
        random_state=0
        )
    return Data(X_train, X_test, y_train, y_test)

@pytest.fixture(scope='module')
def regression_data():
    X, y = make_regression()
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.5, 
        random_state=0
        )
    return Data(X_train, X_test, y_train, y_test)
