from datetime import datetime
from frozendict import frozendict

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import ShuffleSplit

from seal.experiment import Experiment
from seal.utils.utils import get_hash, filter_attributes_model_selection,\
    check_built, get_object_summary, freeze_nested_dictionary
    


kwargs = {
    'hp_space':
        {
            'n_estimators': [20, 30],
            'max_features': ['auto', 'log2'],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8, 10]
        },
    'date':
        datetime(2021, 3, 23, 10, 57, 54, 72210)
}


def test_get_hash():
    assert get_hash(**kwargs) == '7d673aea3d6ea639262a77e123111a03ea0dac7c3ed05d047a3a98d42e640831',\
        'Incorrect hash values'
        
def test_filter_attributes_model_selection():
    test_shuffle_split = ShuffleSplit()
    
    non_default_attributes = filter_attributes_model_selection(
        test_shuffle_split
    )
    assert "_default_test_size" not in non_default_attributes, \
        "_default_test_size should not be in the test dictionary"

def test_check_build_decorator():

    @check_built("test message",True)
    def test_func(expe: Experiment):
        pass
    
    experiment_test = Experiment(
        use_case='use_case_test',
        problem='classification'
    )

    with pytest.raises(AttributeError) as exc_info:
        test_func(experiment_test)
    assert type(exc_info.value) == AttributeError, "AttributeError \
    should be raised"

def test_get_object_summary():

    experiment_test = Experiment('test', 'classification')
    
    X, y = make_classification()
    X = pd.DataFrame(X)
    y = pd.Series(y)

    experiment_test.build(
        X,
        y,
        metric_to_optimize='roc_auc',
        metrics_to_track=None,
        hyperopt_splitting_strategy=ShuffleSplit()
    )

    df_expected = pd.DataFrame({
        'characteristic': [
            'use_case',
            'experiment_name',
            'experiment_id',
            'creation date',
            'problem',
            'X shape',
            'metrics',
            'splitting strategies'
        ],
        'value': [
            experiment_test.use_case,
            experiment_test.experiment_name,
            experiment_test.experiment_id,
            experiment_test.date,
            experiment_test.problem,
            experiment_test.X.shape,
            str(experiment_test.metrics),
            str(experiment_test.splitting_strategy)
        ]
    })

    pd.testing.assert_frame_equal(
        get_object_summary(experiment_test), df_expected)
    with pytest.raises(AttributeError) as exc_info:
        experiment_test.use_case = "changed_use_case"
    assert type(exc_info.value) == AttributeError, "AttributeError \
    should be raised"

def test_freeze_nested_dictionary():
    test_dictionary = {
        "first test dictionary": {
            "test": 1
        },
        "second test dictionary": {
            "test": 2
        }
    }
    
    test_dictionary = freeze_nested_dictionary(test_dictionary)
    
    assert isinstance(
        test_dictionary, frozendict
    ) and isinstance(
        test_dictionary["first test dictionary"], frozendict
    )
