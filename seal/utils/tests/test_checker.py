import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold

from seal.experiment import Experiment
from seal.utils.checker import ExperimentPlanChecker

@pytest.fixture()
def classification_experiment() -> Experiment:
    return Experiment("test", "classification")

def test_experiment_problem(classification_experiment):
    experiment = classification_experiment
    experiment.problem = "test_problem"
    
    with pytest.raises(ValueError) as exc_info:
        ExperimentPlanChecker()._check_experiment_problem(experiment)
    
    assert type(exc_info.value) == ValueError, "Wrong problem type"

def test_wrong_metric_problem(classification_experiment):
    X, y = load_iris(return_X_y=True, as_frame=True)
    
    experiment = classification_experiment
    with pytest.raises(AssertionError) as exc_info:
        experiment.build(
            X,
            y, 
            metric_to_optimize="r2",
            metrics_to_track=None,
            hyperopt_splitting_strategy=KFold(4)
        )
    
    assert type(exc_info.value) == AssertionError, "Wrong metric definition"
