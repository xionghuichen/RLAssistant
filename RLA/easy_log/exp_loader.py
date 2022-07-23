from RLA.easy_log import logger
from RLA.easy_log.tester import exp_manager, Tester
import copy
import argparse
from typing import Optional, OrderedDict, Union, Dict, Any
from RLA.const import DEFAULT_X_NAME
from pprint import pprint

class ExperimentLoader(object):
    """
    We can use a combination of the functions: import_hyper_parameters, load_from_record_date and fork_tester_log_files
    to construct classical tasks.
    -  Start a new task: do nothing.
    -  load a pretrained model for another task (e.g., validation):
        0. config loaded_task_name and loaded_date to the task and timestamp of the target experiment to load respectively.
        1. init your exp_manager;
        2. call exp_loader.load_from_record_date to resume the neural networks and intermediate variables.
        3. start your process.
    - resume an experiment:
        0. config loaded_task_name and loaded_date to the task and timestamp of the target experiment to load respectively.
        1. init your exp_manager;
        2. call exp_loader.fork_tester_log_files to copy all of the log data of the target experiment to the current experiment.
        3. call exp_loader.load_from_record_date to resume the neural networks and intermediate variables.
        4. start your process.
    - resume an experiment with other settings.
        0. config loaded_task_name and loaded_date to the task and timestamp of the target experiment to load respectively.
        1. call exp_loader.load_from_record_date
        2. call import_hyper_parameters to resume the hyper-parameters of the target experiment
            and use hp_to_keep to overwrite the hyper-parameters that you want to update for the new test.
        3. call exp_loader.load_from_record_date to resume the neural networks and intermediate variables.
        4. start your process.
    """
    def __init__(self):
        self.task_name = exp_manager.hyper_param.get('loaded_task_name', None)
        self.load_date = exp_manager.hyper_param.get('loaded_date', None)
        self.data_root = getattr(exp_manager, 'data_root', None)
        if self.data_root is None:
            self.data_root = getattr(exp_manager, 'root', None)
        pass

    def config(self, task_name, record_date, root):
        self.task_name = task_name
        self.load_date = record_date
        self.data_root = root

    @property
    def is_valid_config(self):
        if self.load_date is not None and self.task_name is not None and self.data_root is not None:
            return True
        else:
            logger.warn("meet invalid loader config when use it")
            logger.warn("load_date", self.load_date)
            logger.warn("task_name", self.task_name)
            logger.warn("root", self.data_root)
            return False

    def import_hyper_parameters(self, hp_to_overwrite: Optional[list] = None, sync_timestep=False):
        if self.is_valid_config:
            loaded_tester = Tester.load_tester(self.load_date, self.task_name, self.data_root)
            target_hp = copy.deepcopy(exp_manager.hyper_param)
            target_hp.update(loaded_tester.hyper_param)
            if hp_to_overwrite is not None:
                for v in hp_to_overwrite:
                    target_hp[v] = exp_manager.hyper_param[v]
            args = argparse.Namespace(**target_hp)
            args.load_date = self.load_date
            args.load_task_name = self.task_name
            if sync_timestep:
                load_iter = loaded_tester.get_custom_data(DEFAULT_X_NAME)
                exp_manager.time_step_holder.set_time(load_iter)
            return args
        else:
            return argparse.Namespace(**exp_manager.hyper_param)

    def load_from_record_date(self, var_prefix: Optional[str] = None, variable_list: Optional[list]=None, verbose=True,
                              ckp_index: Optional[int]=None):
        """

        :param var_prefix: the prefix of namescope (for tf) to load. Set to '' to load all of the parameters.
        :param variable_list: the saved variables in the process of training, e.g., data buffer, decayed learning rate.
        :return:
        """
        if self.is_valid_config:
            loaded_tester = Tester.load_tester(self.load_date, self.task_name, self.data_root)
            if verbose:
                print("attrs of the loaded tester")
                pprint(loaded_tester.__dict__)
            # load checkpoint
            load_res = {}
            if var_prefix is not None:
                loaded_tester.new_saver(var_prefix=var_prefix, max_to_keep=1)
                _, load_res = loaded_tester.load_checkpoint(ckp_index)
            else:
                loaded_tester.new_saver(max_to_keep=1)
                _, load_res = loaded_tester.load_checkpoint(ckp_index)
            hist_variables = {}
            if variable_list is not None:
                for v in variable_list:
                    hist_variables[v] = loaded_tester.get_custom_data(v)
            load_iter = loaded_tester.get_custom_data(DEFAULT_X_NAME)
            return load_iter, load_res, hist_variables
        else:
            return 0, {}, {}

    def fork_log_files(self):
        """
        copy the log files in task_name/load_date to the new experiment.
        :return:
        """
        if self.is_valid_config:
            global exp_manager
            assert isinstance(exp_manager, Tester)
            loaded_tester = Tester.load_tester(self.load_date, self.task_name, self.data_root)
            # copy log file
            exp_manager.log_file_copy(loaded_tester)
            # copy attribute
            exp_manager.hyper_param = loaded_tester.hyper_param
            exp_manager.hyper_param_record = loaded_tester.hyper_param_record
            exp_manager.private_config = loaded_tester.private_config


exp_loader = experimental_loader = ExperimentLoader()
