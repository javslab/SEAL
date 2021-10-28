from datetime import datetime
from hashlib import blake2b
import pickle
from typing import Dict, List, Union, TYPE_CHECKING

from frozendict import frozendict
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit
from sklearn.model_selection._split import BaseShuffleSplit
from tabulate import tabulate

from seal.metrics import Metrics
from seal.run import Run
from seal.splitting_strategy import SplittingStrategy
from seal.utils.names import get_random_name
from seal.utils.utils import check_built, filter_attributes_model_selection, get_object_summary, freeze_nested_dictionary
    
if TYPE_CHECKING:
    from seal.components.base import Component


class Experiment(object):
    """
    Experiment is the central object of the SEAL Library.
    
    The main aim of this object is to make some assumptions about a machine \
    learning problem through an experiment plan design phase, please see \
    :meth:`~seal.experiment.Experiment.add` and \
    :meth:`~seal.experiment.Experiment.build`.
    
    Once the experiment plan is well designed, the user has the possibility to \
    find the best model with the execution of his experiment's plan through \
    automl backend engine (https://automl.github.io/auto-sklearn/master/.) \
    and our high level api :meth:`~seal.experiment.Experiment.start_run`.
    
    Parameters
    ----------
    use_case : str
       use case name
    problem : str
        2 options regression or classification
    """

    def __init__(
            self,
            use_case: str,
            problem: str
        ) -> None:

        self.__use_case = use_case
        self.__experiment_name = get_random_name()
        self.__date = datetime.now()
        self.__problem = problem
        
        self.__components = {
            "experiment_components": {},
            "hybrid_components": {},
            "run_components": {}
        }
        
        self.__is_built = False

    @check_built("You cannot add a Component for a built Experiment")
    def add(self, component: "Component") -> None:
        """
        Add a :class:`~seal.components.base.Component`.
        
        This method add a :class:`~seal.components.base.Component` to the \
        current :class:`~seal.base.experiment.Experiment` instance. Note that,
        the added :class:`~seal.components.base.Component` must implement, 
        :class:`~seal.components.base.ExperimentComponent`, \
        :class:`~seal.components.base.RunComponent`, or both of them, see \
        :class:`~seal.components.base.HybridComponent`.
        
        Please also note that:
            - It is not possible to add any other \
                :class:`~seal.components.base.Component` when the \
                :meth:`~seal.experiment.Experiment.build` had been called
            - All added :class:`~seal.components.base.ExperimentComponent` \
                are called, when the :meth:`~seal.experiment.Experiment.build` \
                had been called
            - All added :class:`~seal.components.base.RunComponent` \
                are called, when the  \
                :meth:`~seal.experiment.Experiment.start_run` \
                had been called (after the hyperoptimization iteration)
        
        Parameters
        ----------
        component : :class:`~seal.components.base.Component`
            a Component instance that implement \
            :class:`~seal.components.base.ExperimentComponent`, \
            :class:`~seal.components.base.RunComponent`, or both of them.
        """
        from seal.components.base import ExperimentComponent, \
            HybridComponent, Logger, RunComponent
        
        if isinstance(component, HybridComponent):
            self.__components["hybrid_components"].update(
                        {
                            str(component): component
                        }
                    )
        
        elif isinstance(component, ExperimentComponent):
            self.__components["experiment_components"].update(
                        {
                            str(component): component
                        }
                    )
        
        elif isinstance(component, RunComponent):
            self.__components["run_components"].update(
                        {
                            str(component): component
                        }
                    )
       
        elif isinstance(component, Logger):
            self._logger = component
        
        else:
            raise TypeError(
                "The added component must be an instance of class Component"
            )
                    
    @check_built("You cannot rebuild an Experiment")
    def build(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            metric_to_optimize: Union[str, Dict],
            metrics_to_track: Union[List[str], None],
            hyperopt_splitting_strategy: Union[
                BaseCrossValidator, BaseShuffleSplit
            ],
            sort_by: str = None,
            test_size: Union[float, int] = None,
            **kwargs
    ) -> None:
        """
        Checks the consistency of the user-defined Experiment plan, and build an
        Experiment by executing this plan.

        This method guarantees that all necessary elements of the plan are well
        defined and consistent with each other. Once the plan is checked, it is
        applied using data passed as arguments : splitting strategies for validation
        and hyperparameter optimization, sanity check.
        Once the Experiment is established and certified, it will serve as a
        launching ramp for the auto machine learning runs.

        Parameters
        ----------
        X: pd.DataFrame (n_samples, n_features)
           Input samples. DataFrame that will be used during the Experiment's lifecycle.
           This object will serve as a basis for the Experiment construction
           (cross-validation, sanity checking estimation), and will be also used
           for automl iteration
        y: pd.Series of shape (n_samples, 1)
           The target variable (corresponding to X) that will be used during
           the Experiment's lifecycle.
        metric_to_optimize: Union[str, Dict]
            str or dict that will be mapped to a 
            :class:`~autosklearn.metrics.Scorer`, this metric will be used for 
            solving the machine learning problem.
        metrics_to_track: Union[List[str], None]
            list of str that will be mapped to 
            :dict:`autosklearn.metrics.CLASSIFICATION_METRICS` or 
            :dict:`autosklearn.metrics.REGRESSION_METRICS`, in order to track 
            additional metrics, not required can be NONE
        hyperopt_splitting_strategy: Union[BaseCrossValidator, BaseShuffleSplit]
            cross-validation object used for hyperopt split
        sort_by: str
            when hyperopt_splitting_strategy is a TimeSeriesSplit, \
            indicates by which column the dataset have to be sorted.
            Not required can be NONE
        test_size: Union[float, int], optional
            represent the proportion of the dataset to include in the test split
            will be passed to \
            :func:`sklearn.model_selection.train_test_split` for train/test split
        kwargs: dict
            other parameters and their values that will be passed
            to :func:`sklearn.model_selection.train_test_split` for train/test split
        """
        from seal.utils.checker import ExperimentPlanChecker

        self.__splitting_strategy = SplittingStrategy(
            hyperopt_splitting_strategy=hyperopt_splitting_strategy,
            sort_by=sort_by,
            test_size=test_size,
            **kwargs
        )

        self.__metrics = Metrics(
            metric_to_optimize=metric_to_optimize, metrics_to_track=metrics_to_track
        )

        self.__base_index = X.index.tolist()
        self.__X, self.__y = self.__splitting_strategy(X, y)

        ExperimentPlanChecker().run_checks(self)

        self.__data_id = blake2b(
            pd.util.hash_pandas_object(
                pd.concat([self.__X, self.__y], axis=1)
            ).values, digest_size=5
        ).hexdigest()

        self.__splitting_strategies_id = blake2b(
            (
                str(self.splitting_strategy.hyperopt_splitting_strategy)+\
                str(self.splitting_strategy.train_test_split_arguments)
            ).encode('utf-8'),
            digest_size=5
        ).hexdigest()

        self.__call_components(self)
        
        self.__is_built = True
        
        self.__components = freeze_nested_dictionary(self.__components)
        
        if hasattr(self, "_logger"):
            self._logger.log_experiment(self)
    
    def __call_components(self, object: Union["Experiment", "Run"]) -> None:
        
        if isinstance(object, Experiment):
            targeted_components = "experiment_components"
            
        elif isinstance(object, Run):
            targeted_components = "run_components"
        
        components_to_call = {
            **self.__components["hybrid_components"],
            **self.__components[targeted_components]
        }

        list(
            map(
                lambda component: component(object), components_to_call.values()
            )
        )
        
    @staticmethod
    def load(path: str) -> 'Experiment':
        """
        Load an instance of :class:`~seal.experiment.Experiment`
 
        Parameters
        ----------
        path: str 
            path to load the :class:`~seal.experiment.Experiment`
 
        Returns
        -------
        experiment: :class:`~seal.experiment.Experiment`
            instance of :class:`~seal.experiment.Experiment`
        """
 
        with open(path, 'rb') as file:
            experiment = pickle.load(file)
 
        return experiment

    @check_built(
        "You can't start a Run for an unbuilt Experiment", True
    )
    def start_run(
            self,
            **kwargs
            ) -> Run:
        """
        Create and start a :class:`~seal.run.Run`.
        
        According to the :class:`~seal.experiment.Experiment` plan this \
        method create and start an hyperoptimization cycle through \
        :class:`~seal.run.Run` and our automl backend engine \
        https://automl.github.io/auto-sklearn/master/.
        
        Parameters
        ----------
        kwargs: dict
            specification of hyperoptimization space according to our automl \
            backend engine, please see:
             - https://automl.github.io/auto-sklearn/master/api.html#classification
             - https://automl.github.io/auto-sklearn/master/api.html#regression
            
        Returns
        -------
        run : :class:`~seal.run.Run`
             instance of :class:`~seal.run.Run`
        """           
        
        kwargs.update(
            {
                "metric": self.metrics.metric_to_optimize_scorer,
                "resampling_strategy": 
            self.splitting_strategy.hyperopt_splitting_strategy.__class__,
                "resampling_strategy_arguments":
            filter_attributes_model_selection(
                self.splitting_strategy.hyperopt_splitting_strategy
                )
            }
        )
        
        run_parameters = {
            "problem": self.problem,
            "data_train": {
                "X": self.X.iloc[self.splitting_strategy.train_index], 
                "y": self.y.iloc[self.splitting_strategy.train_index]
            },
            "data_test": {
                "X": self.X.iloc[self.splitting_strategy.test_index], 
                "y": self.y.iloc[self.splitting_strategy.test_index]
            },
            "hp_rules": kwargs,
            "metrics": self.metrics
        }
        
        run = Run(**run_parameters)

        self.__call_components(run)
        
        if hasattr(self, "_logger"):
            self._logger.log_run(run)
        
        return run

    @check_built("You cannot summarize an unbuilt Experiment", True)
    def summary(self) -> None:
        """
        Display a summary of an instance of \
        :class:`~seal.experiment.Experiment` 
            
        This method describe the :class:`~seal.experiment.Experiment` 
        plan through all its characteristics and all its potential \
        :class:`~seal.components.base.Component`
            
        See :func:`~seal.utils.utils.get_object_summary`
        """
        title = 'Experiment summary : \n\n'

        content = tabulate(
            get_object_summary(self),
            showindex=False, headers="keys"
        )

        print(title + content)

    @property
    def base_index(self) -> List[int]:
        return self.__base_index

    @property
    def components(self) -> frozendict:
        return self.__components

    @property
    def date(self) -> datetime:
        return self.__date

    @property
    def experiment_id(self) -> str:
        return "{}_{}".format(self.__data_id, self.__splitting_strategies_id)

    @property
    def experiment_name(self) -> str:
        return self.__experiment_name

    @property
    def is_built(self) -> bool:
        return self.__is_built

    @property
    def metrics(self) -> 'Metrics':
        return self.__metrics

    @property
    def problem(self) -> str:
        return self.__problem

    @problem.setter
    @check_built("You cannot change the problem for a built Experiment")
    def problem(self, value: str) -> None:
        self.__problem = value
    
    @property
    def splitting_strategy(self) -> 'SplittingStrategy':
        return self.__splitting_strategy

    @property
    def use_case(self) -> str:
        return self.__use_case

    @use_case.setter
    @check_built("You cannot change the use_case name for a built Experiment")
    def use_case(self, value: str) -> None:
        self.__use_case = value

    @property
    def X(self) -> pd.DataFrame:
        return self.__X

    @property
    def y(self) -> pd.Series:
        return self.__y
