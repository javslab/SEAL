import pytest

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import TimeSeriesSplit, \
    ShuffleSplit

from seal.splitting_strategy import SplittingStrategy

@pytest.fixture
def get_splitting_strategy() -> SplittingStrategy:
    return SplittingStrategy(TimeSeriesSplit(), sort_by='0')

@pytest.fixture
def get_splitting_strategy_called(get_splitting_strategy) -> SplittingStrategy:
    X, y = make_classification()
    X = pd.DataFrame(X)
    y = pd.Series(y)

    X.rename(columns={0: "0"}, inplace=True)

    X, y = get_splitting_strategy(X,y)

    return {
        'split_strat': get_splitting_strategy,
        'X': X,
        'y': y
    }

@pytest.mark.parametrize(
    "splitting_strategies,parameters", [
        (SplittingStrategy, (ShuffleSplit, {'test_size': 0.2})),
        (SplittingStrategy, (TimeSeriesSplit(), 0.2, 2))
    ]
)
def test_wrong_init_parameters_type(splitting_strategies, parameters):
    with pytest.raises(TypeError) as exc_info:
        splitting_strategies(*parameters) 

    assert type(exc_info.value) == TypeError, "Wrong parameter type"

def test_missing_sort_by():
    with pytest.raises(KeyError) as exc_info:
        SplittingStrategy(TimeSeriesSplit())

    assert type(exc_info.value) == KeyError, "Missing sort_by parameter"

def test_get_hyperopt_splitting_strategy(get_splitting_strategy):
    assert hasattr(get_splitting_strategy, 'hyperopt_splitting_strategy'), \
        "SplittingStrategy instance should have a hyperopt_splitting_strategy \
        attribute"

def test_get_train_test_split_arguments(get_splitting_strategy):
    assert hasattr(get_splitting_strategy, 'train_test_split_arguments'), \
        "SplittingStrategy instance should have a train_test_split_arguments \
        attribute"

def test_get_hyperopt_splitting_strategy(get_splitting_strategy_called):
    assert hasattr(get_splitting_strategy_called['split_strat'], 'train_index'), \
        "SplittingStrategy instance should have a train_index attribute"

def test_get_hyperopt_splitting_strategy(get_splitting_strategy_called):
    assert hasattr(get_splitting_strategy_called['split_strat'], 'test_index'), \
        "SplittingStrategy instance should have a test_index attribute"

def test_get_hyperopt_splitting_strategy(get_splitting_strategy_called):
    assert hasattr(get_splitting_strategy_called['split_strat'], 'cv_index'), \
        "SplittingStrategy instance should have a cv_index attribute"

def test_X_sorted(get_splitting_strategy_called):
    assert get_splitting_strategy_called['X']['0'].is_monotonic, "X not \
    sorted"

def test_str(get_splitting_strategy):
    expected_value = "For hyperparameter optimization : TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)\nFor validation (train/test split) : {'test_size': None}"

    assert str(get_splitting_strategy) == expected_value, \
        "__str__ returns wrong value"