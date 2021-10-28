from inspect import signature
from typing import Dict, List, Union 

from autosklearn.metrics import make_scorer, CLASSIFICATION_METRICS, \
    REGRESSION_METRICS
import numpy as np

ALL_METRICS = {
    **CLASSIFICATION_METRICS,
    **REGRESSION_METRICS
}


class Metrics(object):
    """
    Metrics is a mandatory parameter of :class:`~seal.experiment.Experiment`.
    
    This object leaves you the possibility to define two types of metrics at
    the experimentation building time (see :meth:`~seal.experiment.Experiment.build` for more details about a built experiment)
        - `metric_to_optimize`: metric use for solving the machine learning problem
        - `metrics_to_track`: a set of metrics for tracking additional metrics
    
    Once the experiment is built, the user-defined metrics are passed though a 
    :class:`~seal.run.Run`. Once the :class:`~seal.run.Run` is fitted, 
    metrics are available under the :meth:`~seal.run.Run.metrics` property
    
    Note that, all available metrics and metrics computation relies
    entirely on the autosklearn implementation, see \
        `autosklearn <https://automl.github.io/auto-sklearn/master/api.html#built-in-metrics>_`
    
    Parameters
    ----------
    metric_to_optimize : Union[str, Dict]
        name of the metric to optimize among the autosklearn\
            implementation or dict that will be mapped to a \
        :class:`autosklearn.metrics.Scorer`, this metric will be used for \
        solving the machine learning problem. See :class:`autosklearn.metrics.make_scorer` and\
        `autosklearn <https://automl.github.io/auto-sklearn/master/api.html#built-in-metrics>_` 
    metrics_to_track : List[str], optional
        list of str that will be mapped to \
        :const:`autosklearn.metrics.CLASSIFICATION_METRICS` or \
        :const:`autosklearn.metrics.REGRESSION_METRICS`, in order to track \
        additional metrics, by default None
    """
    def __init__(self,
                 metric_to_optimize: Union[str, Dict],
                 metrics_to_track: Union[List[str], None] = None
    ) -> None:
        if metrics_to_track:
            if isinstance(metrics_to_track, List):
                self.experiment_metrics = {
                    metric_name: metric_function._score_func
                    for metric_name, metric_function in ALL_METRICS.items()
                    if metric_name in metrics_to_track
                }
            else:
                raise TypeError(
                    "metrics_to_track must be either `None` or a list of str"
                )
        
        else:
            self.experiment_metrics = dict()

        if isinstance(metric_to_optimize, str):
            if metric_to_optimize in ALL_METRICS.keys():
                self.metric_to_optimize_scorer = ALL_METRICS[
                    metric_to_optimize
                ]
                
                self.experiment_metrics.update({
                            metric_to_optimize: 
                            ALL_METRICS[metric_to_optimize]._score_func
                        })
            else:
                raise KeyError(
                    f"Unknown metric {metric_to_optimize}, please see\
                        refer to the object documentation"
                )

        elif isinstance(metric_to_optimize, Dict):
            self.metric_to_optimize_scorer = make_scorer(**metric_to_optimize)

            self.experiment_metrics.update({
                    self.metric_to_optimize_scorer.name:
                    self.metric_to_optimize_scorer._score_func
                })
        
        else:
            raise TypeError(
                "metric_to_optimize must be str or dict"
            )

        self.__metrics_with_score = [
            metric_name for metric_name, metric_function in self.experiment_metrics.items()
            if 'y_score' in signature(metric_function).parameters.keys()
        ]

    def __call__(
            self,
            y_true: np.ndarray,
            y_pred: Union[None, np.ndarray] = None,
            y_score: Union[None, np.ndarray] = None
        ) -> Dict[str, float]:
        """
        Compute user defined metrics

        Compute all metrics specified in the current instance. The computed \
        metrics are returned as a dictionary following this template \
        {metric_name: metric_value}

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Ground truth (correct) labels (classification) / values (regression)
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted labels (classification) / Estimated target values \
            (regression) by default None
        y_score : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target scores, by default None, see also \
            `scikit-learn metrics documentation \
            <https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics>`__

        Returns
        -------
        Dict[str, float]
            Dictionary following this template \
            {metric_name: metric_value}
        """
        return self.__compute_metrics(y_true, y_pred, y_score)

    def __compute_metrics(
            self,
            y_true: np.ndarray,
            y_pred: Union[None, np.ndarray] = None,
            y_score: Union[None, np.ndarray] = None
        ) -> Dict[str, float]:

        metrics = dict()
        for metric_name, metric_function in self.experiment_metrics.items():
            if metric_name in self.__metrics_with_score:
                if y_score is not None:
                # The output of the `predict_proba()` is a 2d array, \
                    # however the roc_auc_score function requires a 1d array \
                        # we only keep the second column of the array \
                            # What about the case where y_score.shape[-1] > 2??`
                    if y_score.shape[-1] == 2:
                        y_score = y_score[:, 1]
                    metrics[metric_name] = \
                        metric_function(y_true, y_score)
                else:
                    raise ValueError(
                        f"Missing parameter `y_score` for metric :{metric_name}"
                    )
            elif y_pred is not None:
                metrics[metric_name] = \
                    metric_function(y_true, y_pred)
            else:
                raise ValueError(
                    f"Missing parameter `y_pred` for metric: {metric_name}"
                )

        return metrics

    def __str__(self) -> str:
        metric_to_optimize = 'To optimize : {}'.format(
            self.metric_to_optimize_scorer
        )
        metrics_to_track = '\nTo track : {}'.format(
            ', '.join(
                list(self.experiment_metrics.keys())
            )
        )
        
        return metric_to_optimize + metrics_to_track