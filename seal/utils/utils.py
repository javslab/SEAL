from datetime import datetime
from functools import wraps
import json
from hashlib import sha256
from inspect import signature
from typing import Callable, Dict, TYPE_CHECKING, Union

import pandas as pd
from frozendict import frozendict
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import BaseShuffleSplit

if TYPE_CHECKING:
    from seal.experiment import Experiment


def get_hash(
        **kwargs
) -> str:
    """ Return a hash of parameters """
    
    hash = sha256()
    for key, value in kwargs.items():
        if isinstance(value, datetime):
            hash.update(str(kwargs[key]).encode('utf-8'))
        else:
            hash.update(json.dumps(kwargs[key]).encode())

    return hash.hexdigest()

def filter_attributes_model_selection(
    obj: Union[BaseCrossValidator, BaseShuffleSplit]
    ) -> Dict:
    """
    filter_attributes_model_selectiion filter all non-essentials attributes \
    in a BaseCrossValidator or BaseShuffleSplit object.
    
    Parameters
    ----------
    obj : Union[BaseCrossValidator, BaseShuffleSplit] \
    BaseCrossValidator or BaseShuffleSplit object used \
    in :class:`seal.components.splitting_strategy.SplittingStrategy`

    Returns
    -------
    Dict
        dict with all necessary attributes for re-instantiate \
        BaseCrossValidator or BaseShuffleSplit object
    """
    
    if not isinstance(obj, (BaseCrossValidator, BaseShuffleSplit)):
        raise ValueError(
            "obj should be an instance of BaseCrossValidator / BaseShuffleSplit"
        )
    
    return dict(
        (p, obj.__dict__[p]) for p in signature(obj.__class__).parameters
    )

def check_built(message: str, need_build: bool = False) ->  Callable:
    """
    check_built is a decorator used for methods that must be called on \
    built or unbuilt :class:`~seal.experiment.Experiment`.
    If the :class:`~seal.experiment.Experiment` is_built attribute has \
    not the correct value, an AttributeError is raised with the message passed \
    as argument.

    Parameters
    ----------
    message: str
        Error message
    need_build: bool
        Expected value for :class:`~seal.experiment.Experiment` is_built \
        attribute
    
    Returns
    -------
    Callable
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(
            experiment: 'Experiment',
            *args,
            **kwargs
        ) -> Callable:
            if experiment.is_built == need_build:
                return func(experiment, *args, **kwargs)
            else:
                raise AttributeError(message)
        return wrapper
    return decorator

def get_object_summary(experiment: 'Experiment') -> pd.DataFrame:
    """
    Builds and returns a dataframe containing characteristics of an instance of
    :class:`~seal.experiment.Experiment` and its potential
    :class:`~seal.components.base.Component`
    """

    characteristics = [
        ('use_case', experiment.use_case),
        ('experiment_name', experiment.experiment_name),
        ('experiment_id', experiment.experiment_id),
        ('creation date', experiment.date),
        ('problem', experiment.problem),
        ('X shape', experiment.X.shape),
        ('metrics', str(experiment.metrics)),
        ('splitting strategies', str(experiment.splitting_strategy)),
    ]
    
    for component_type in experiment.components:
        if len(experiment.components[component_type]):
            for component in experiment.components[component_type]:
                characteristics.append((
                    component+' component characteristics',
                    str(
                        experiment.components[component_type][component]
                    )
                ))
    
    return pd.DataFrame(characteristics, columns=["characteristic", "value"])

def freeze_nested_dictionary(nested_dictonary: Dict) -> frozendict:
    """
    freeze_nested_dictionary is a function that freezes a nested dictionary

    Parameters
    ----------
    nested_dictonary : Dict
        The nested dictionary to freeze

    Returns
    -------
    frozendict
        The frozen nested dictionary
    """
    for key, values in nested_dictonary.items():
        nested_dictonary[key] = frozendict(**values)
        
    nested_dictonary = frozendict(**nested_dictonary)

    return nested_dictonary
