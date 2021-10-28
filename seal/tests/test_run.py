import pytest

from seal.run import Run
from seal.metrics import Metrics

@pytest.fixture
def get_run_classification(classification_data):
    run_parameters = {
        'problem':'classification',
        'hp_rules' : {
                "time_left_for_this_task": 30
            },
        'data_train': {
                'X': classification_data.X_train,
                'y': classification_data.y_train
            },
        'data_test': {
                'X': classification_data.X_test,
                'y':classification_data.y_test
            },
        "metrics": Metrics(
            metric_to_optimize="roc_auc", 
            metrics_to_track=["accuracy", "f1"]
            )
    }
    return Run(**run_parameters)

@pytest.fixture
def get_run_regression(regression_data):
    run_parameters = {
            'problem':'regression',
            'hp_rules' : {
                    "time_left_for_this_task": 30
                },
            'data_train': {
                'X': regression_data.X_train,
                'y': regression_data.y_train
            },
            'data_test': {
                'X': regression_data.X_test,
                'y': regression_data.y_test
            },
            'metrics': Metrics(
                metric_to_optimize='mean_absolute_error', 
                metrics_to_track=['r2']
                )
        }
    return Run(**run_parameters)

def test_run_classification_model(get_run_classification):
    assert hasattr(get_run_classification, 'model'), "Run instance should \
        have a model attribute"

def test_run_classification_metrics(get_run_classification):
    assert len(get_run_classification.metrics['test']) == 3 and \
        len(get_run_classification.metrics['train']) == 3 , 'Incorrect number of metrics'
        
def test_run_classification_run_id(get_run_classification):
    assert hasattr(get_run_classification, 'run_id'), "Run instance should \
        have a run_id attribute"

def test_run_regression_model(get_run_regression):
    assert hasattr(get_run_regression, 'model'), "Run instance should have a\
        model attribute"
        
def test_run_regression_metrics(get_run_regression):
    assert len(get_run_regression.metrics['test']) == 2 and \
        len(get_run_regression.metrics['train']) == 2, 'Incorrect number of metrics'

def test_run_regression_run_id(get_run_regression):
    assert hasattr(get_run_regression, 'run_id'), "Run instance should have a\
        run_id attribute"
        