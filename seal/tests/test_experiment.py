from io import StringIO
import os
import pickle
import sys

import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, TimeSeriesSplit
from tabulate import tabulate

from seal.components.base import ExperimentComponent, HybridComponent, \
RunComponent
from seal.components.file_system_logger import FileSystemLogger
from seal.experiment import Experiment
from seal.run import Run
from seal.utils.utils import get_object_summary


class TestBadComponent():
    pass


class TestExperimentComponent(ExperimentComponent):
    def _call_on_experiment(self, object: "Experiment") -> None:
        self.called = 1
    
    def __str__(self) -> str:
        return "TEC"


class TestRunComponent(RunComponent):
    def _call_on_run(self, object: "Run") -> None:
        self.called = 1
    
    def __str__(self) -> str:
        return "TRC"
      
        
class TestHybridComponent(HybridComponent):
    def _call_on_experiment(self, object: "Experiment") -> None:
        self.called_on_experiment = 1
        
    def _call_on_run(self, object: "Run") -> None:
        self.called_on_run = 1
        
    def __str__(self) -> str:
        return "THC"


@pytest.fixture
def build_classification_experiment(classification_experiment):
    X, y = load_iris(return_X_y=True, as_frame=True)
    experiment = classification_experiment
    experiment.add(TestExperimentComponent())
    experiment.add(TestHybridComponent())
    experiment.add(TestRunComponent())
    experiment.build(
        X,
        y, 
        metric_to_optimize="f1_micro",
        metrics_to_track=None,
        hyperopt_splitting_strategy=TimeSeriesSplit(),
        sort_by=X.columns[0]
    )
    return experiment

def test_date_property(classification_experiment):
    assert classification_experiment.date, "Missing date attribute"

def test_experiment_name_property(classification_experiment):
    assert classification_experiment.experiment_name, "Missing name attribute" 

def test_is_built_property(classification_experiment):
    assert classification_experiment.is_built == False, "bad is_built value"

def test_problem_property(classification_experiment):
    assert classification_experiment.problem == "classification", "bad pb type"

def test_use_case_property(classification_experiment):
    assert classification_experiment.use_case == "test", "bad use case name"

@pytest.mark.parametrize(
    ["component", "component_signature" ,"component_key"],
    [
        (TestExperimentComponent(), "TEC", "experiment_components"),
        (TestHybridComponent(), "THC", "hybrid_components"),
        (TestRunComponent(), "TRC", "run_components")
    ]
)
def test_add_component(
    classification_experiment, component, component_signature, component_key
):
    experiment = classification_experiment
    experiment.add(component)
    assert component_signature in experiment.components[component_key].keys()

def test_add_logger(classification_experiment):
    experiment = classification_experiment
    experiment.add(FileSystemLogger("."))
    assert hasattr(experiment, "_logger"), "Logger is not properly set"

def test_add_bad_component(classification_experiment):
    experiment = classification_experiment
    with pytest.raises(TypeError) as exc_info:
        experiment.add(TestBadComponent())
    assert type(exc_info.value) == TypeError, "Wrong error for bad component"

def test_experiment_id(build_classification_experiment):
    assert build_classification_experiment.experiment_id, "Missing experiment\
        id attribute"

def test_experiment_call_experiment_component(build_classification_experiment):
    assert hasattr(
        build_classification_experiment.components["experiment_components"]["TEC"], "called"
    ), "Experiment Component not called"

def test_experiment_call_hybrid_component(build_classification_experiment):
    assert hasattr(
        build_classification_experiment.components["hybrid_components"]["THC"], "called_on_experiment"
    ), "Hybrid Component not called on Experiment"

def test_experiment_is_built(build_classification_experiment):
    assert build_classification_experiment.is_built == True, "bad is_built \
    value"

def test_experiment_metrics(
  build_classification_experiment  
):
    assert hasattr(
        build_classification_experiment,
        "metrics"
    ), "Missing metric attribute"
    
def test_experiment_splitting_strategy_property(
    build_classification_experiment
):
    assert hasattr(
        build_classification_experiment,
        "splitting_strategy"
    ), "Mssing splitting_strategy attribute"

def test_experiment_X_property(build_classification_experiment):
    assert hasattr(
        build_classification_experiment,
        "X"
    ), "Missing X attribute"
   
def test_experiment_y_property(build_classification_experiment):
    assert hasattr(
        build_classification_experiment,
        "y"
    ), "Missing y attribute"
    
def test_start_run_classification(build_classification_experiment):
    run_test = build_classification_experiment.start_run(
        time_left_for_this_task = 60
    )
    assert isinstance(run_test, Run), "start_run should return an Run instance"

def test_start_run_call_run_component(build_classification_experiment):
    run_test = build_classification_experiment.start_run(
        time_left_for_this_task = 60
    )
    assert hasattr(
        build_classification_experiment.components["run_components"]["TRC"],
        "called"
    ), "Run Component not called"

def test_start_run_call_hybrid_component(build_classification_experiment):
    run_test = build_classification_experiment.start_run(
        time_left_for_this_task = 60
    )
    assert hasattr(
        build_classification_experiment.components["hybrid_components"]["THC"],
        "called_on_run"
    ), "Hybrid Component not called on Run"

# To move -> test_file_system_logger.py
def test_log_experiment(classification_experiment):
    X, y = load_iris(return_X_y=True, as_frame=True)

    experiment = classification_experiment
    experiment.add(FileSystemLogger("."))
    
    experiment.build(
        X,
        y, 
        metric_to_optimize="f1_micro",
        metrics_to_track=None,
        hyperopt_splitting_strategy=KFold(4)
    )
    
    path_to_check = os.path.join(
        experiment.use_case, experiment.experiment_name, "experiment.pkl"
    )
    
    assert os.path.exists(path_to_check), f"{path_to_check} doesn't exist"

def test_load_experiment(build_classification_experiment):
    with open("experiment.pkl", "wb") as file:
        pickle.dump(build_classification_experiment, file)
    
    experiment = Experiment.load("experiment.pkl")
    assert isinstance(experiment, Experiment), "Experiment should be \
    an instance of Experiment"

def test_experiment_summary(build_classification_experiment):
    result = StringIO()
    sys.stdout = result
    build_classification_experiment.summary()

    title = 'Experiment summary : \n\n'

    content = tabulate(
        get_object_summary(build_classification_experiment),
        showindex=False, headers="keys"
    )
    
    assert result.getvalue() == title + content + '\n',\
        'summary method returns incorrect value'