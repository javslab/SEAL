from autosklearn.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS
from sklearn.utils.validation import check_X_y

from seal.experiment import Experiment

ALL_METRICS = {
    **CLASSIFICATION_METRICS,
    **REGRESSION_METRICS
}


class ExperimentPlanChecker(object): 
    """
    ExperimentPlanChecker is an object that checks the experiment plan.
    
    At the :meth:`~seal.experiment.Experiment.build` moment, this object \
    run severals checks in order to see if the experiment plan is well designed.
    
    Here is an overview of the checks performed by the object:
        - :meth:`~seal.utils.checker.ExperimentPlanChecker._check_arrays`\
        : see whether X and y attribute are compliant with \
        sklearn standards.
        - :meth:`~seal.utils.checker.ExperimentPlanChecker._check_experiment_problem`: see if the problem type is correctly \
        informed by the user.
        - :meth:`~seal.utils.checker.ExperimentPlanChecker._check_problem_metrics`: see if the known metrics are consistent with \
        the experiment problem
    """
                  
    def _check_arrays(self, experiment: Experiment) -> None:
        _, _ = check_X_y(experiment.X, experiment.y)
    
    def _check_experiment_problem(self, experiment: Experiment) -> None:
        if not experiment.problem in ["classification", "regression"]:
            raise ValueError(
                f"Unknown problem: {experiment.problem}, please see documentation"
            )
    
    def _check_problem_metrics(self, experiment: Experiment) -> None:
        if experiment.problem == "classification":
            targeted_metrics = CLASSIFICATION_METRICS
        else:
            targeted_metrics = REGRESSION_METRICS
        
        for metric in experiment.metrics.experiment_metrics.keys():
            if metric in ALL_METRICS: # Only check for known metrics
                assert metric in targeted_metrics, f"Metric: {metric} is inconsistent with problem: {experiment.problem}"
    
    def run_checks(self, experiment: Experiment) -> None:
        """
        Perform some tests on the experiment plan
        
        Several checks are performed in order to check if the
        experiment plan is consistent:
            - checks the experiment problem 
            - checks the metrics provided by the user
            - checks the data provided by the user (scikit learn wrapper)
        
        Parameters
        ----------
        experiment : :class:`~seal.experiment.Experiment`
            an Experiment instance
        """
        self._check_experiment_problem(experiment)
        self._check_problem_metrics(experiment)
        self._check_arrays(experiment)
