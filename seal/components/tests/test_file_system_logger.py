import os

import pytest
from sklearn.model_selection import KFold

from seal.experiment import Experiment 
from seal.components.file_system_logger import FileSystemLogger


@pytest.fixture
def build_classification_experiment(get_classification_data):
    experiment = Experiment(
        use_case='test_use_case',
        problem='classification'
    )
    experiment.add(FileSystemLogger('.'))
    experiment.build(
            get_classification_data.X,
            get_classification_data.y,
            metric_to_optimize='roc_auc',
            metrics_to_track=['recall', 'precision'],
            hyperopt_splitting_strategy=KFold(5)       
    )
    return experiment

def test_is_logged_experiment(build_classification_experiment):
    path_to_file= os.path.join(
        ".", 
        build_classification_experiment.use_case, 
        build_classification_experiment.experiment_name,
        'experiment.pkl'
    )
    assert os.path.isfile(path_to_file)
    
def test_is_logged_model(build_classification_experiment):
    run_for_test = build_classification_experiment.start_run(
        time_left_for_this_task=30
    )
    path_to_model = os.path.join(
        '.', 
        build_classification_experiment.use_case, 
        build_classification_experiment.experiment_name,
        run_for_test.run_id,
        'model.pkl'
    )
    assert os.path.isfile(path_to_model)
    
def test_is_logged_params(build_classification_experiment):
    run_for_test = build_classification_experiment.start_run(
        time_left_for_this_task=30
    )
    path_to_hp_parameters = os.path.join(
        '.', 
        build_classification_experiment.use_case, 
        build_classification_experiment.experiment_name,
        run_for_test.run_id,
        'hp_parameters.json'
    )
    path_to_models_parameters = os.path.join(
        '.', 
        build_classification_experiment.use_case, 
        build_classification_experiment.experiment_name,
        run_for_test.run_id,
        'models_parameters.txt'       
    )
    assert os.path.isfile(path_to_hp_parameters) and os.path.isfile(path_to_models_parameters)
    
def test_is_logged_metrics(build_classification_experiment):
    run_for_test = build_classification_experiment.start_run(
        time_left_for_this_task=30
    ) 
    path_to_metrics = os.path.join(
        '.', 
        build_classification_experiment.use_case, 
        build_classification_experiment.experiment_name,
        run_for_test.run_id,
        'metrics.json'
    )
    assert os.path.isfile(path_to_metrics)
    
def test_is_logged_experiment_existing_use_case(get_classification_data): 
    experiment = Experiment(
        use_case='test_use_case', 
        problem='classification'
    )

    experiment.add(FileSystemLogger('.'))
    experiment.build(
            get_classification_data.X,
            get_classification_data.y,
            metric_to_optimize='roc_auc',
            metrics_to_track=['recall', 'precision'],
            hyperopt_splitting_strategy=KFold(5)       
    )
    path_to_file= os.path.join(
            ".", 
            experiment.use_case, 
            experiment.experiment_name,
            'experiment.pkl'
    )
    assert os.path.isfile(path_to_file)
