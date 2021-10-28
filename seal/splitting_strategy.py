import logging
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit, \
    train_test_split
from sklearn.model_selection._split import BaseShuffleSplit

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class SplittingStrategy(object):
    """
    SplittingStrategy is a mandatory parameter of :class:`~seal.experiment.Experiment`.

    This object lets you define two splitting strategies at the experimentation
    building time (see :meth:`~seal.experiment.Experiment.build` for more
    details about a built experiment) :
        - how to split the data into train and test subsets, by defining at \
        least the test size
        - a cross-validator used for hyperparameter optimization
    
    Once the experiment is built, the generated indices are availaible under the
    :attr:`~seal.experiment.Experiment.splitting_strategy.train_index`,
    :attr:`~seal.experiment.Experiment.splitting_strategy.test_index` and
    :attr:`~seal.experiment.Experiment.splitting_strategy.cv_index`,
    properties.

    Parameters
    ----------
    hyperopt_splitting_strategy: Union[BaseCrossValidator, BaseShuffleSplit]
        cross-validation object used for hyperopt split
    sort_by: str, optional
        when hyperopt_splitting_strategy is a TimeSeriesSplit, \
        indicates by which column the dataset have to be sorted
    test_size: Union[float, int], optional
        represent the proportion of the dataset to include in the test split
        will be passed to \
        :func:`sklearn.model_selection.train_test_split` for train/test split
    kwargs: dict
        other parameters and their values that will be passed
        to :func:`sklearn.model_selection.train_test_split` for train/test split
    """

    def __init__(
        self,
        hyperopt_splitting_strategy: Union[BaseCrossValidator, BaseShuffleSplit],
        sort_by: str = None,
        test_size: Union[float, int] = None,
        **kwargs
    ) -> None:

        if isinstance(hyperopt_splitting_strategy, (BaseCrossValidator, BaseShuffleSplit)):
            if isinstance(hyperopt_splitting_strategy, TimeSeriesSplit):
                if sort_by is not None:
                    if isinstance(sort_by, str):
                        self.__sort_by = sort_by
                    else:
                        raise TypeError("sort_by must be str")
                else:
                    raise KeyError("When hyperopt_splitting_strategy is a \
                    TimeSeriesSplit, you must provide sort_by argument")
                
                if "shuffle" not in kwargs or kwargs["shuffle"]:
                    logger.warning("Data will be randomly splitted into train \
                        and test subsets, whereas hyperopt_splitting_strategy \
                        is a TimeSeriesSplit cross-validator. If you don't \
                        want to, use shuffle=False")

            self.__hyperopt_splitting_strategy = hyperopt_splitting_strategy

            kwargs.update({"test_size": test_size})
            self.__train_test_split_arguments = kwargs

        else:
            raise TypeError("hyperopt_splitting_strategy must be an instance of BaseCrossValidator or BaseShuffleSplit")
    
    def __call__(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """
        Set up the validation and hyperopt splitting strategies (and their
        arguments)
        This method will be executed when calling
        :meth:`~seal.experiment.Experiment.build`

        Parameters
        ----------
        X: pd.DataFrame (n_samples, n_features)
            Input samples. DataFrame that will be used during the Run's lifecycle.
        y: pd.Series of shape (n_samples, 1)
           The target variable (corresponding to X)
        """

        if hasattr(self, "sort_by"):
            X = X.sort_values(by=[self.__sort_by])
            y = y[X.index]
        
        self.__train_index, self.__test_index = train_test_split(
            X.index,
            **self.__train_test_split_arguments
        )

        self.__cv_index = [
            (
                self.__train_index[train_cv],
                self.__train_index[val_cv]
            ) for train_cv, val_cv in self.__hyperopt_splitting_strategy.split(
                X.iloc[self.__train_index],
                y.iloc[self.__train_index]
            )
        ]

        return X, y
    
    def __str__(self) -> str:
        splitting_strategies ='For hyperparameter optimization : {}'\
            .format(self.hyperopt_splitting_strategy)
        splitting_strategies = '{}\nFor validation (train/test split) : {}'\
            .format(splitting_strategies, self.train_test_split_arguments)
        return splitting_strategies
    
    @property
    def cv_index(self) -> List[Tuple[np.ndarray]]:
        return self.__cv_index

    @property
    def hyperopt_splitting_strategy(self) -> Union[BaseCrossValidator, BaseShuffleSplit]:
        return self.__hyperopt_splitting_strategy
    
    @property
    def sort_by(self) -> str:
        return self.__sort_by
    
    @property
    def test_index(self) -> np.ndarray:
        return self.__test_index

    @property
    def train_index(self) -> np.ndarray:
        return self.__train_index
    
    @property
    def train_test_split_arguments(self) -> BaseShuffleSplit:
        return self.__train_test_split_arguments
