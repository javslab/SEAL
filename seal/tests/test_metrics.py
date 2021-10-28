from typing import Dict, Tuple

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from seal.metrics import Metrics


def get_data_classification() -> Tuple[np.ndarray]:
    y_true = np.random.randint(2, size=(20,))
    y_score = np.random.uniform(0.01, 0.99, size=(20,))
    y_pred = np.where(y_score > 0.5, 1, 0)
    return y_true, y_pred, y_score

def get_data_regression() -> Tuple[np.ndarray]:
    y_true = np.random.normal(size=(20,))
    y_pred = np.random.normal(size=(20,))
    return y_true, y_pred

@pytest.mark.parametrize(
    "metrics,data", [
        (
            Metrics("f1_micro", ["accuracy", "roc_auc"]), get_data_classification()
        ),
        (
            Metrics("r2", ["mean_squared_error", "mean_absolute_error"]),
            get_data_regression()
        )
    ]
)
def test_metrics_computation(metrics, data):
    metrics_result = metrics(*data)
    assert isinstance(metrics_result, Dict), \
        "metrics_result must be an instance of dict"

def test_metrics_custom_scorer():
    custom_scorer = {
        "name": "custom_auc",
        "score_func": roc_auc_score
    }

    metrics = Metrics(
        metric_to_optimize=custom_scorer,
        metrics_to_track=["f1_micro", "accuracy"]
    )
    
    results = metrics(*get_data_classification())
    
    assert "custom_auc" in results, "results does not contain custom metric"

def test_unknown_metric():
    with pytest.raises(KeyError) as exc_info:
        metrics = Metrics("test_metric", ["accuracy", "f1_micro"])
    
    assert type(exc_info.value) == KeyError, "Unknown metric"

@pytest.mark.parametrize(
    "metrics,parameters", [
        (Metrics, ("f1_micro", {"accuracy": "test"})),
        (Metrics, (["f1_mico"], ["accuracy"]))
    ]
)
def test_metrics_wrong_init_parameters(metrics, parameters):
    with pytest.raises(TypeError) as exc_info:
        test_metrics = metrics(*parameters) 

    assert type(exc_info.value) == TypeError, "Wrong parameter type"

def test_missing_y_score():
    metrics = Metrics("roc_auc")
    y_true, y_pred, _ = get_data_classification()
    
    with pytest.raises(ValueError) as exc_info:
        results = metrics(y_true, y_pred)
        
    assert type(exc_info.value) == ValueError, "Missing y_score parameter"

def test_missing_y_pred():
    metrics = Metrics("f1_micro")
    y_true, _, y_score = get_data_classification()
    
    with pytest.raises(ValueError) as exc_info:
        results = metrics(y_true, None, y_score)
        
    assert type(exc_info.value) == ValueError, "Missing y_pred parameter"

def test_metrics_str():
    metrics = Metrics(
        metric_to_optimize="roc_auc",
        metrics_to_track=["f1_micro", "roc_auc"]
    )

    assert str(metrics) == 'To optimize : roc_auc\nTo track : roc_auc, f1_micro', \
        "__str__ returns wrong value"