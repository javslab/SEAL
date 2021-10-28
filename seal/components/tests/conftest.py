from collections import namedtuple

import pandas as pd
import pytest
from sklearn.datasets import make_classification

Data = namedtuple("data", ['X','y'])

@pytest.fixture(scope='module')
def get_classification_data():
    X, y = make_classification()
    return Data(pd.DataFrame(X), pd.Series(y))
