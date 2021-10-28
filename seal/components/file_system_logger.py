import json
import logging
import os
import pickle
from typing import TYPE_CHECKING

from seal.components.base import Logger

if TYPE_CHECKING:
    from seal.experiment import Experiment
    from seal.run import Run

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)


class FileSystemLogger(Logger):
    """
    FileSystemLogger is a subclass of :class:`~seal.components.base.Logger`.
    
    It is a particular component since it is called at the end of both \
    :class:`~seal.experiment.Experiment`and :class:`~seal.run.Run`.
    
    When added to :class:`~seal.experiment.Experiment`, this component \
    automatically saves generated materials created during the \
    :class:`seal.base.experiment.Experiment` life-cycle. It is not a  \
    mandatory component. 
    
    Two levels of backup are handled by this object:
        - During the call of :meth:`~seal.experiment.Experiment.build()`
        - During the call of :meth:`~seal.experiment.Experiment.start_run()`
        
    If no implementation of :class:`seal.components.base.Logger` is \
    specified as a component, the management of the generated materials \
    is under the user responsability.

    Parameters
    ----------
    path : str
        root path or directory, from which will be saved artifacts and metadata 
    """

    def __init__(self, path: str) -> None:
        self.__path = path
        
    def log_experiment(self, experiment: 'Experiment') -> None:
        """
        log_experiment perfoms the first level of backup as described 
        in the object description. 
        
        This method creates the needed folders and saves an instance of \
        :class:`~seal.experiment.Experiment`.

        Parameters
        ----------
        experiment: :class:`seal.base.experiment.Experiment`
            an instance of Experiment
        """
        self.__use_case_path = os.path.join(self.__path, 
                                           experiment.use_case)
        self.__experiment_path = os.path.join(self.__use_case_path, 
                                             experiment.experiment_name)

        if not os.path.exists(self.__use_case_path):
            logger.info(
                "No {} folder found, creating {} folders".format(
                    experiment.use_case,
                    self.__experiment_path
                )
            )
            os.makedirs(self.__experiment_path)
        else:
            logger.info(
                "{} folder found, creating {} folder".format(
                    experiment.use_case,
                    self.__experiment_path
                )
            )
            os.mkdir(self.__experiment_path)

        artifact_name = os.path.join(self.__experiment_path,
                                     "experiment.pkl")
                                        
        with open(artifact_name, "wb") as output_file:
            logger.info(
                "Experiment instance saved in {}".format(artifact_name)
            )
            pickle.dump(experiment, output_file)
            
    def log_run(self, run: 'Run') -> None:
        """
        log_run is called at the end of a hyperopt cycle and allow to save the \
        following elements:
            - model: Best model found, available under the \
                :class:`~seal.run.Run.model` attribute
            - models_parameters : Best models parameters found, availabe when \
                calling the :meth:`~seal.run.Run.model.get_models_with_weights()` method
            - hp_parameters: Best hyperopt parameters found, available when\
                calling the :meth:`~seal.run.Run.model.get_params()` method
            - metrics: train and test metrics for the best model found
            
        This implementation relies on the current seal backend engine, \
        autosklearn. 
        """
        
        self.__run_id_path = os.path.join(self.__experiment_path, 
                                          run.run_id)
        
        if not os.path.exists(self.__run_id_path):
            logger.info(
                f"Creating a run folder: {self.__run_id_path}"
            )
            os.mkdir(self.__run_id_path)
            
        model_path = os.path.join(
            self.__run_id_path, 
            'model.pkl'
        )
        hp_parameters_path = os.path.join(
            self.__run_id_path,
            'hp_parameters.json'
        )
        models_parameters_path = os.path.join(
            self.__run_id_path,
            'models_parameters.txt'
        )
        metrics_path = os.path.join(
            self.__run_id_path,
            'metrics.json'
        )
        
        self._log_model(run, model_path)
        self._log_params(run, hp_parameters_path, models_parameters_path)
        self._log_metrics(run, metrics_path) 
    
    def _log_metrics(self, run: 'Run', path: str) -> None:
        with open(path, 'w') as output_file:
            logger.info(
                "Metrics saved in {}".format(path)
            )
            json.dump(run.metrics, output_file, indent=4)
            
    def _log_model(self, run: 'Run', path: str) -> None:        
        with open(path, 'wb') as output_file:
            logger.info(
                "Model saved in {}".format(path)
            )
            pickle.dump(run.model, output_file)

    def _log_params(self, 
                    run: 'Run', 
                    hp_parameters_path: str, 
                    models_parameters_path: str) -> None:      
        with open(hp_parameters_path, 'w') as output_file:
            logger.info(
                "Hp parameters saved in {}".format(
                    hp_parameters_path
                )
            )
            hp_parameters = {
                param:str(value) for param,value in run.model.get_params().items()
            }
            json.dump(hp_parameters, output_file, indent=4)

        with open(models_parameters_path, 'w') as output_file:
            logger.info(
                "Model's parameters saved in {}".format(
                    models_parameters_path
                )
            )
            output_file.write(
                '\n'.join(
                    str(item) for item in run.model.get_models_with_weights()
                ) 
            )
            
