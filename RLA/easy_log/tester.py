#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017-11-12
# Modified    :   2017-11-12
# Version     :   1.0
from collections import deque
import dill
import copy
import time
import os

import json
import datetime
import os.path as osp
import pprint
import numpy as np


from RLA.easy_log.time_step import time_step_holder
from RLA.easy_log import logger
from RLA.easy_log.const import *
from RLA.const import *
import yaml
import shutil
import argparse
from typing import Dict, List, Tuple, Type, Union, Optional
from RLA.utils.utils import deprecated_alias, load_yaml
from RLA.const import DEFAULT_X_NAME, FRAMEWORK
import pathspec

def import_hyper_parameters(task_table_name, record_date):
    """
    return the hyper parameters of the experiment in task_table_name/record_date, which is stored in Tester.

    :param task_table_name:
    :param record_date:
    :return:
    """
    logger.warn("the function is deprecated. please check the ExperimentLoader as the new implementation")
    global tester
    assert isinstance(tester, Tester)
    load_tester = tester.load_tester(record_date, task_table_name, tester.data_root)

    args = argparse.Namespace(**load_tester.hyper_param)
    return args


def load_from_record_date(task_table_name, record_date):
    """
    load the checkpoint of the experiment in task_table_name/record_date.
    :param task_table_name:
    :param record_date:
    :return:
    """
    logger.warn("the function is deprecated. please check the ExperimentLoader as the new implementation")
    global tester
    assert isinstance(tester, Tester)
    load_tester = tester.load_tester(record_date, task_table_name, tester.data_root)
    # load checkpoint
    load_tester.new_saver(var_prefix='', max_to_keep=1)
    load_iter, load_res = load_tester.load_checkpoint()
    tester.time_step_holder.set_time(load_iter)
    tester.print_log_dir()
    return load_iter, load_res


def fork_tester_log_files(task_table_name, record_date):
    """
    copy the log files in task_table_name/record_date to the new experiment.
    :param task_table_name:
    :param record_date:
    :return:
    """
    logger.warn("the function is deprecated. please check the ExperimentLoader as the new implementation")
    global tester
    assert isinstance(tester, Tester)
    load_tester = tester.load_tester(record_date, task_table_name, tester.data_root)
    # copy log file
    tester.log_file_copy(load_tester)
    # copy attribute
    tester.hyper_param = load_tester.hyper_param
    tester.hyper_param_record = load_tester.hyper_param_record
    tester.private_config = load_tester.private_config


class Tester(object,):

    def __init__(self):
        self.__custom_recorder = {}
        self.__ipaddr = None
        self.custom_data = {}
        self.time_step_holder = time_step_holder
        self.hyper_param = {}
        self.strftims = None
        self.private_config = None
        self.last_record_fph_time = None
        self.hyper_param_record = []
        self.metadata_list = []
        self.summary_add_dict = {}
        self._rc_start_time = {}
        self.pkl_dir = None
        self.checkpoint_dir = None
        self.pkl_file = None
        self.results_dir = None
        self.log_dir = None
        self.code_dir = None
        self.saver = None
        self.dl_framework = None
        self.checkpoint_keep_list = None
        self.log_name_format_version = LOG_NAME_FORMAT_VERSION.V1

    @deprecated_alias(task_name='task_table_name', private_config_path='rla_config', log_root='data_root')
    def configure(self, task_table_name: str, rla_config: Union[str, dict], data_root: str,
                  ignore_file_path: Optional[str] = None, run_file: Union[str, List[str]] = None,
                  is_master_node: bool = False, code_root: Optional[str] = None):
        """
        The function to configure your exp_manager, which should be run before your experiments.
        :param task_table_name: define a ``table'' to store a collection of experiment data item.
        :type task_table_name: str
        :param rla_config: Pass the location of rla_config.yaml. It defines all of the running strategies of RLA.
        Ref to RLAssistant/example/rla_config.yaml
        :type rla_config: str
        :param data_root: define the location of the RLA database.
        :type data_root: str
        :param ignore_file_path: RLA will backup the codebase of each experiment (defined in rla_config.yaml).
         If there are some files unnecessary to backup,
         you can customize the pattern of files to ignore with the same rules of gitignore (https://git-scm.com/docs/gitignore).
         We recommend you to pass the location of .gitignore directly to ignore_file_path.
        :type ignore_file_path: str
        :param run_file: If you have extra files out of your codebase (e.g., some scripts to run the code), you can pass it to the run_file.
        Then we will backup the run_file too.
        :type run_file: str or list
        :param is_master_node: In "distributed training & centralized logs" mode (By set SEND_LOG_FILE in rla_config.yaml to True),
        you should mark the master node (is_master_node=True) to collect logs of the slave nodes (is_master_node=False).
        :type is_master_node: bool
        :param code_root:  Define the root of your codebase (for backup) explicitly. It will be in the same location as rla_config.yaml by default.
        """
        if isinstance(rla_config, str):
            self.private_config = load_yaml(rla_config)
        elif isinstance(rla_config, dict):
            self.private_config = rla_config
        else:
            raise NotImplementedError
        self.run_file = run_file
        self.ignore_file_path = ignore_file_path
        self.task_table_name = task_table_name
        self.data_root = data_root
        logger.info("private_config: ")
        self.dl_framework = self.private_config["DL_FRAMEWORK"]
        self.is_master_node = is_master_node

        if code_root is None:
            if isinstance(rla_config, str):
                self.project_root = "/".join(rla_config.split("/")[:-1])
            else:
                raise NotImplementedError("If you pass the rla_config dict directly, "
                                          "you should define the root of your codebase (for backup) explicitly by pass the code_root.")
        else:
            self.project_root = code_root
        for k, v in self.private_config.items():
            logger.info("k: {}, v: {}".format(k, v))

    def get_task_table_name(self):
        task_table_name = getattr(self, 'task_table_name', None)
        if task_table_name is None:
            task_table_name = getattr(self, 'task_name', None)
            print("[WARN] you are using an old-version RLA. "
                  "Some attributes' name have been changed (task_name->task_table_name).")
            if task_table_name is None:
                raise RuntimeError("invalid ExpManager: task_table_name cannot be found", )
        return task_table_name

    def set_hyper_param(self, **argkw):
        """
        This method is to record all of hyper parameters to test object.

        Place pass your parameters as follow format:
            self.set_hyper_param(param_a=a,param_b=b) or a dict self.set_hyper_param(**{'param_a'=a,'param_b'=b})

        Note: It is invalid to pass a local object to this function.

        Parameters
        ----------
        argkw : key-value
            for example: self.set_hyper_param(param_a=a,param_b=b)

        """
        self.hyper_param = argkw

    def update_hyper_param(self, k, v):
        self.hyper_param[k] = v

    def clear_record_param(self):
        self.hyper_param_record = []

    def log_files_gen(self):
        info = None
        self.record_date = datetime.datetime.now()
        logger.info("gen log files for record date : {}".format(self.record_date))
        if info is None:
            info = self.auto_parse_info()
            info = '&' + info
        self.info = info
        code_dir, _ = self.__create_file_directory(osp.join(self.data_root, CODE, self.task_table_name), '', is_file=False)
        log_dir, _ = self.__create_file_directory(osp.join(self.data_root, LOG, self.task_table_name), '', is_file=False)
        self.pkl_dir, self.pkl_file = self.__create_file_directory(osp.join(self.data_root, ARCHIVE_TESTER, self.task_table_name), '.pkl')
        self.checkpoint_dir, _ = self.__create_file_directory(osp.join(self.data_root, CHECKPOINT, self.task_table_name), is_file=False)
        self.results_dir, _ = self.__create_file_directory(osp.join(self.data_root, OTHER_RESULTS, self.task_table_name), is_file=False)
        self.log_dir = log_dir
        self.code_dir = code_dir

        self._init_logger()
        self.serialize_object_and_save()
        self.__copy_source_code(self.run_file, code_dir)
        self._feed_hyper_params_to_tb()
        self.print_log_dir()

    def update_log_files_location(self, root:str):
        """
        This function is designed for the requirement of using copied/moved experiment logs to other databases for downstream task.
        The location of the experiment logs might have changed compared with their original location.
        The function automatically update the attributes related to the data_root to the current location.
        :param root: current data_root
        :type root: str
        """
        self.data_root = root

        task_table_name = self.get_task_table_name()
        code_dir, _ = self.__create_file_directory(osp.join(self.data_root, CODE, task_table_name), '', is_file=False)
        log_dir, _ = self.__create_file_directory(osp.join(self.data_root, LOG, task_table_name), '', is_file=False)
        self.pkl_dir, self.pkl_file = self.__create_file_directory(osp.join(self.data_root, ARCHIVE_TESTER, task_table_name), '.pkl')
        self.checkpoint_dir, _ = self.__create_file_directory(osp.join(self.data_root, CHECKPOINT, task_table_name), is_file=False)
        self.results_dir, _ = self.__create_file_directory(osp.join(self.data_root, OTHER_RESULTS, task_table_name), is_file=False)
        self.log_dir = log_dir
        self.code_dir = code_dir
        self.print_log_dir()

    def _init_logger(self):
        self.writer = None
        # logger configure
        logger.info("store file %s" % self.pkl_file)
        logger.configure(self.log_dir, self.private_config["LOG_USED"], framework=self.private_config["DL_FRAMEWORK"])
        for fmt in logger.Logger.CURRENT.output_formats:
            if isinstance(fmt, logger.TensorBoardOutputFormat):
                self.writer = fmt.writer
        if "tensorboard" not in self.private_config["LOG_USED"]:
            time_step_holder.config(0, 0, tf_log=False)

    def log_file_copy(self, source_tester):
        assert isinstance(source_tester, Tester)
        shutil.rmtree(self.checkpoint_dir)
        shutil.copytree(source_tester.checkpoint_dir, self.checkpoint_dir)
        if os.path.exists(source_tester.results_dir):
            shutil.rmtree(self.results_dir)
            shutil.copytree(source_tester.results_dir, self.results_dir)
        else:
            logger.warn("[load warning]: can not find results dir")
        if os.path.exists(source_tester.log_dir):
            shutil.rmtree(self.log_dir)
            shutil.copytree(source_tester.log_dir, self.log_dir)
        else:
            logger.warn("[load warning]: can not find log dir")
        self._init_logger()

    def task_gen(self, task_pattern_list):
        return '-'.join(task_pattern_list)

    def print_log_dir(self):
        logger.info("log dir: {}".format(self.log_dir))
        logger.info("pkl_file: {}".format(self.pkl_file))
        logger.info("checkpoint_dir: {}".format(self.checkpoint_dir))
        logger.info("results_dir: {}".format(self.results_dir))

    @classmethod
    def load_tester(cls, record_date, task_table_name, log_root):
        logger.info("load tester")
        res_dir, res_file = cls.log_file_finder(record_date, task_table_name=task_table_name,
                                                file_root=osp.join(log_root, ARCHIVE_TESTER),
                                                log_type='files')
        import dill
        load_tester = dill.load(open(osp.join(res_dir, res_file), 'rb'))
        assert isinstance(load_tester, Tester)
        logger.info("update log files' root")
        load_tester.update_log_files_location(root=log_root)
        logger.info("load data: \n ts {}, \n ip {}, \n info {}".format(
            str(load_tester.record_date.strftime("%Y/%m/%d")) + '/' + load_tester.record_date_to_str(
                load_tester.record_date), load_tester.ipaddr, load_tester.info))



        return load_tester

    def add_record_param(self, keys):
        for k in keys:
            if '.' in k:
                try:
                    sub_k_list = k.split('.')
                    v = self.hyper_param[sub_k_list[0]]
                    for sub_k in sub_k_list[1:]:
                        v = v[sub_k]
                    self.hyper_param_record.append(str(k) + '=' + str(v).replace('[', '{').replace(']', '}').replace('/', '_'))
                except KeyError as e:
                    print("do not include dot ('.') in your hyperparemeter name")
            else:
                self.hyper_param_record.append(str(k) + '=' + str(self.hyper_param[k]).replace('[', '{').replace(']', '}').replace('/', '_'))

    def add_summary_to_logger(self, summary, name='', simple_val=False, freq=20):
        """
        [deprecated] see RLA.logger.log_from_tf_summary
        """
        logger.warn("add_summary_to_logger is deprecated. See RLA.logger.log_from_tf_summary.")
        if "tensorboard" not in self.private_config["LOG_USED"]:
            logger.info("skip adding summary to tb")
            return
        if name not in self.summary_add_dict:
            self.summary_add_dict[name] = []
        if freq > 0:
            summary_ts = int(self.time_step_holder.get_time() / freq)
        else:
            summary_ts = 0
        if freq <= 0 or summary_ts not in self.summary_add_dict[name]:
            from tensorflow.core.framework import summary_pb2
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            if simple_val:
                list_field = summ.ListFields()

                def recursion_util(inp_field):
                    if hasattr(inp_field, "__getitem__"):
                        for inp in inp_field:
                            recursion_util(inp)
                    elif hasattr(inp_field, 'simple_value'):
                        logger.record_tabular(name + '/' + inp_field.tag, inp_field.simple_value)
                    else:
                        pass
                recursion_util(list_field)
                logger.dump_tabular()
            else:
                self.writer.add_summary(summary, self.time_step_holder.get_time())
                self.writer.flush()
            self.summary_add_dict[name].append(summary_ts)

    def _feed_hyper_params_to_tb(self, metric_dict=None):
        if "tensorboard" not in self.private_config["LOG_USED"]:
            logger.info("skip feeding hyper-param to tb")
            return
        for fmt in logger.Logger.CURRENT.output_formats:
            if isinstance(fmt, logger.TensorBoardOutputFormat):
                fmt.add_hyper_params_to_tb(self.hyper_param, metric_dict)

    def sync_log_file(self, skip_error=False):
        """
        syn_log_file is an automatic synchronization function.
        It will send all log files (e.g., code/**, checkpoint/**, log/**, etc.) to your target server via the FTP protocol.
        To run this function, you should add some configuration on rla_config.yaml.
        We transfer files in SFTP by setting cnopts.hostkeys=None for convenience which is not safe.

        SEND_LOG_FILE: boolean. denotes synchronization or not.
        ftp_server: target server ip address
        username: username of target server
        password: password of target server
        remote_porject_dir: log root of target server, e.g., "/Project/SRG/SRG/var_gan_imitation/"

        :param skip_error: if skip_error==True, we will skip the error of sync.
        :type skip_error: bool
        """

        logger.warn("sync: start")
        remote_data_root = self.private_config["REMOTE_SETTING"].get("remote_data_root")
        if remote_data_root is None:
            remote_data_root = self.private_config["REMOTE_SETTING"].get("remote_log_root")
            logger.warn("the parameter remote_log_root will be renamed to remote_data_root in future versions.")
        else:
            raise RuntimeError("miss remote_log_root in rla_config")

        def send_data(ftp_obj):
            for root, dirs, files in os.walk(self.log_dir):
                suffix = root.split("/{}/".format(LOG))
                assert len(suffix) == 2, "root should only have one pattern \"/log/\""
                remote_root = osp.join(remote_data_root, LOG, suffix[1])
                local_root = root
                logger.warn("sync {} <- {}".format(remote_root, local_root))
                for file in files:
                    ftp_obj.upload_file(remote_root, local_root, file)

        if self.private_config["SEND_LOG_FILE"] and not self.is_master_node:
            from RLA.auto_ftp import ftp_factory
            alternative_protocol = 'ftp'
            try:
                if 'file_transfer_protocol' not in self.private_config["REMOTE_SETTING"].keys():
                    self.private_config["REMOTE_SETTING"]['file_transfer_protocol'] = 'ftp'
                ftp = ftp_factory(name=self.private_config["REMOTE_SETTING"]['file_transfer_protocol'],
                                  server=self.private_config["REMOTE_SETTING"]["ftp_server"],
                                  username=self.private_config["REMOTE_SETTING"]["username"],
                                   password=self.private_config["REMOTE_SETTING"]["password"],
                                   port=self.private_config["REMOTE_SETTING"]["port"])
                if self.private_config["REMOTE_SETTING"]['file_transfer_protocol'] == 'ftp':
                    alternative_protocol = 'sftp'
                else:
                    alternative_protocol = 'ftp'
                send_data(ftp_obj=ftp)
                logger.warn("sync: send success!")
            except Exception as e:
                try:
                    logger.warn("failed to send log files through {}: {} ".format(self.private_config["REMOTE_SETTING"]['file_transfer_protocol'], e))
                    logger.warn("try another protocol:", alternative_protocol)
                    ftp = ftp_factory(name=alternative_protocol,
                                      server=self.private_config["REMOTE_SETTING"]["ftp_server"],
                                      username=self.private_config["REMOTE_SETTING"]["username"],
                                      password=self.private_config["REMOTE_SETTING"]["password"],
                                      port=self.private_config["REMOTE_SETTING"]["port"])
                    send_data(ftp_obj=ftp)
                    logger.warn("sync: send success!")
                except Exception as e:
                    logger.warn("fail to send log files through {}: {} ".format(alternative_protocol, e))

                    logger.warn("server info ftp_server {}, username {}, password {}, remote_data_root {}".format(
                        self.private_config["REMOTE_SETTING"]["ftp_server"],
                        self.private_config["REMOTE_SETTING"]["username"],
                        self.private_config["REMOTE_SETTING"]["password"], remote_data_root))
                    import traceback
                    logger.warn(traceback.format_exc())
                    if not skip_error:
                        raise RuntimeError("fail to sync")
        else:
            logger.warn("skip the sync process.")

    @classmethod
    def log_file_finder(cls, record_date, task_table_name='train', file_root='../checkpoint/', log_type='dir'):
        record_date = datetime.datetime.strptime(record_date, '%Y/%m/%d/%H-%M-%S-%f')
        prefix = osp.join(file_root, task_table_name)
        directory = str(record_date.strftime("%Y/%m/%d"))
        directory = osp.join(prefix, directory)
        file_found = ''
        for root, dirs, files in os.walk(directory):
            if log_type == 'dir':
                search_list = dirs
            elif log_type =='files':
                search_list = files
            else:
                raise NotImplementedError
            for search_item in search_list:
                if search_item.startswith(str(record_date.strftime("%H-%M-%S-%f"))):

                    # self.__ipaddr = split_dir[1]
                    # if version_num is None:
                    #     split_dir = search_item.split(' ')
                    #     info = " ".join(split_dir[2:])
                    #     logger.info("load data: \n ts {}, \n ip {}, \n info {}".format(split_dir[0], split_dir[1], info))
                    #
                    # elif version_num == LOG_NAME_FORMAT_VERSION.V1:
                    #     split_dir = search_item.split('_')
                    #     info = " ".join(split_dir[2:])
                    #     logger.info("load data: \n ts {}, \n ip {}, \n info {}".format(split_dir[0], split_dir[1], info))
                    #
                    # else:
                    #     raise RuntimeError("unknown version name", version_num)

                    file_found = search_item
                    break
        return directory, file_found

    @property
    def ipaddr(self):
        if self.__ipaddr is None:
            self.__ipaddr = self.__gen_ip()
        return self.__ipaddr

    def __gen_ip(self):
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("1.1.1.1", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception as e:
            ip = 'noip'
        return ip

    def get_ignore_files(self, src, names):
        if self.ignore_file_path is None:
            return []
        with open(self.ignore_file_path) as f:
            lines = f.readlines()
            ret_lines = []
            for i in range(len(lines)):
                line = lines[i].strip()
                if '#' in line or '' == line:
                    continue
                ret_lines.append(line)

        spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ret_lines)
        paths = []
        for name in names:
            paths.append(osp.join(src, name))
        match_paths = list(set(spec.match_files(paths)))
        match_names = []
        for idx, path in enumerate(paths):
            if path in match_paths:
                match_names.append(names[idx])
        return match_names


    def __copy_source_code(self, run_file, code_dir):
        import shutil
        def _copy_run_file(run_file, code_dir):
            if type(run_file) == list:
                for file_name in run_file:
                    shutil.copy(file_name, code_dir)
            else:
                shutil.copy(run_file, code_dir)        
        if self.private_config["PROJECT_TYPE"]["backup_code_by"] == 'lib':
            assert os.listdir(code_dir) == []
            os.removedirs(code_dir)
            shutil.copytree(osp.join(self.project_root, self.private_config["BACKUP_CONFIG"]["lib_dir"]), code_dir)
            assert run_file is not None, "you should define the run_file in lib backup mode."
            _copy_run_file(run_file, code_dir)
        elif self.private_config["PROJECT_TYPE"]["backup_code_by"] == 'source':
            if self.private_config["BACKUP_CONFIG"].get("backup_code_dir"):
                for dir_name in self.private_config["BACKUP_CONFIG"]["backup_code_dir"]:
                    shutil.copytree(osp.join(self.project_root, dir_name), osp.join(code_dir, dir_name),
                                    ignore=self.get_ignore_files)
            if run_file is not None:
                _copy_run_file(run_file, code_dir)
        else:
            raise NotImplementedError

    def log_name_formatter(self, prefix, record_date):
        """
        return a unified and unique name for the experiment log.
        :param prefix: prefix location to store the log data.
        :param record_date: the timestamp of the experiment log.
        :return: a unify and unique name
        """
        version_num = self.get_version_num()
        if version_num is None:
            name_format = '{prefix}/{date}/{timestep} {ip} {info}'
        elif version_num == LOG_NAME_FORMAT_VERSION.V1:
            name_format = '{prefix}/{date}/{timestep}_{ip}_{info}'
        else:
            raise RuntimeError("unknown version name", version_num)
        date = record_date.strftime("%Y/%m/%d")
        return name_format.format(prefix=prefix, date=date, timestep=self.record_date_to_str(record_date),
                                                             ip=str(self.ipaddr), info=self.info)

    def record_date_to_str(self, record_date):
        return str(record_date.strftime("%H-%M-%S-%f"))

    def get_version_num(self):
        version_num = getattr(self, 'log_name_format_version', None)
        return version_num

    def __create_file_directory(self, prefix, ext='', is_file=True, record_date=None):
        if record_date is None:
            record_date = self.record_date
        name = self.log_name_formatter(prefix, record_date)
        directory = str(record_date.strftime("%Y/%m/%d"))
        directory = osp.join(prefix, directory)
        if is_file:
            os.makedirs(directory, exist_ok=True)
            file_name = name + ext
        else:
            directory = name + '/'
            os.makedirs(directory, exist_ok=True)
            file_name = ''
        return directory, file_name

    def update_fph(self, cum_epochs):
        if self.last_record_fph_time is None:
            self.last_record_fph_time = time.time()
        else:
            cur_time = time.time()
            duration = (cur_time - self.last_record_fph_time) / 60 / 60
            fph = cum_epochs / duration
            logger.record_tabular('fph', fph)
            # self.last_record_fph_time = cur_time
            logger.dump_tabular()

    def time_record(self, name:str):
        """
        [deprecated] see RLA.easy_log.time_used_recorder
        record the consumed time of your code snippet. call this function to start a recorder.
        "name" is identifier to distinguish different recorder and record different snippets at the same time.
        call time_record_end to end a recorder.
        :param name: identifier of your code snippet.
        :type name: str
        :return:
        :rtype:
        """
        assert name not in self._rc_start_time
        self._rc_start_time[name] = time.time()

    def time_record_end(self, name:str):
        """
        [deprecated] see RLA.easy_log.time_used_recorder
        record the consumed time of your code snippet. call this function to start a recorder.
        "name" is identifier to distinguish different recorder and record different snippets at the same time.
        call time_record_end to end a recorder.
        :param name: identifier of your code snippet.
        :type name: str
        :return:
        :rtype:
        """
        end_time = time.time()
        start_time = self._rc_start_time[name]
        logger.record_tabular("time_used/{}".format(name), end_time - start_time)
        logger.info("[test] func {0} time used {1:.2f}".format(name, end_time - start_time))
        del self._rc_start_time[name]

    # Saver manger.
    def new_saver(self, max_to_keep, var_prefix=None):
        """
        initialize new tf.Saver
        :param var_prefix: we use var_prefix to filter the variables for saving.
        :param max_to_keep:
        :return:
        """
        if self.dl_framework == FRAMEWORK.tensorflow:
            import tensorflow as tf
            if var_prefix is None:
                var_prefix = ''
            try:
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, var_prefix)
                logger.info("save variable :")
                for v in var_list:
                    logger.info(v)
                self.saver = tf.train.Saver(var_list=var_list, max_to_keep=max_to_keep, filename=self.checkpoint_dir,
                                            save_relative_paths=True)

            except AttributeError as e:
                self.max_to_keep = max_to_keep
                # tf.compat.v1.disable_eager_execution()
                # tf = tf.compat.v1
                # var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, var_prefix)
        elif self.dl_framework == FRAMEWORK.torch:
            self.max_to_keep = max_to_keep
        else:
            raise NotImplementedError

    def save_checkpoint(self, model_dict: Optional[dict] = None, related_variable: Optional[dict] = None):
        if self.dl_framework == FRAMEWORK.tensorflow:
            import tensorflow as tf
            iter = self.time_step_holder.get_time()
            cpt_name = osp.join(self.checkpoint_dir, 'checkpoint')
            logger.info("save checkpoint to ", cpt_name, iter)
            try:
                self.saver.save(tf.get_default_session(), cpt_name, global_step=iter)
            except AttributeError as e:
                if model_dict is None:
                    logger.warn("call save_checkpoints without passing a model_dict")
                    return
                if self.checkpoint_keep_list is None:
                    self.checkpoint_keep_list = []
                iter = self.time_step_holder.get_time()
                # tf.compat.v1.disable_eager_execution()
                # tf = tf.compat.v1
                # self.saver.save(tf.get_default_session(), cpt_name, global_step=iter)

                tf.train.Checkpoint(**model_dict).save(tester.checkpoint_dir + "checkpoint-{}".format(iter))
                self.checkpoint_keep_list.append(iter)
                self.checkpoint_keep_list = self.checkpoint_keep_list[-1 * self.max_to_keep:]
        elif self.dl_framework == FRAMEWORK.torch:
            import torch
            if self.checkpoint_keep_list is None:
                self.checkpoint_keep_list = []
            iter = self.time_step_holder.get_time()
            torch.save(model_dict, f=tester.checkpoint_dir + "checkpoint-{}.pt".format(iter))
            self.checkpoint_keep_list.append(iter)
            if len(self.checkpoint_keep_list) > self.max_to_keep:
                for i in range(len(self.checkpoint_keep_list) - self.max_to_keep):
                    rm_ckp_name = tester.checkpoint_dir + "checkpoint-{}.pt".format(self.checkpoint_keep_list[i])
                    logger.info("rm the older checkpoint", rm_ckp_name)
                    os.remove(rm_ckp_name)
                self.checkpoint_keep_list = self.checkpoint_keep_list[-1 * self.max_to_keep:]
        else:
            raise NotImplementedError
        if related_variable is not None:
            for k, v in related_variable.items():
                self.add_custom_data(k, v, type(v), mode='replace')
        self.add_custom_data(DEFAULT_X_NAME, time_step_holder.get_time(), int, mode='replace')
        self.serialize_object_and_save()

    def load_checkpoint(self, ckp_index=None):
        if self.dl_framework == FRAMEWORK.tensorflow:
            # TODO: load with variable scope.
            import tensorflow as tf
            cpt_name = osp.join(self.checkpoint_dir)
            logger.info("load checkpoint {}".format(cpt_name))
            if ckp_index is None:
                ckpt_path = tf.train.latest_checkpoint(cpt_name)
            else:
                ckpt_path = tf.train.latest_checkpoint(cpt_name, ckp_index)
            logger.info("load ckpt_path {}".format(ckpt_path))
            self.saver.restore(tf.get_default_session(), ckpt_path)
            max_iter = ckpt_path.split('-')[-1]
            return int(max_iter), None
        elif self.dl_framework == FRAMEWORK.torch:
            import torch
            all_ckps = os.listdir(self.checkpoint_dir)
            ites = []
            for ckps in all_ckps:
                print("ckps", ckps)
                ites.append(int(ckps.split('checkpoint-')[1].split('.pt')[0]))
            idx = np.argsort(ites)
            all_ckps = np.array(all_ckps)[idx]
            print("all checkpoints:")
            pprint.pprint(all_ckps)
            if ckp_index is None:
                ckp_index = all_ckps[-1].split('checkpoint-')[1].split('.pt')[0]
            return ckp_index, torch.load(self.checkpoint_dir + "checkpoint-{}.pt".format(ckp_index))

    def auto_parse_info(self):
        return '&'.join(self.hyper_param_record)


    def add_graph(self, sess):
        assert self.writer is not None
        self.writer.add_graph(sess.graph)

    def add_custom_data(self, key, data, dtype=list, max_len=-1, mode='append'):
        if mode == 'replace':
            self.custom_data[key] = data
        elif mode == 'append':
            if key not in self.custom_data:
                if issubclass(dtype, deque):
                    assert max_len > 0

                    self.custom_data[key] = deque(maxlen=max_len)
                    self.custom_data[key].append(data)
                elif issubclass(dtype, list):
                    self.custom_data[key] = [data]
                else:
                    self.custom_data[key] = data
            else:
                if issubclass(dtype, list) or issubclass(dtype, deque):
                    self.custom_data[key].append(data)
                else:
                    self.custom_data[key] = data
        else:
            raise NotImplementedError

    def print_custom_data(self, key, prefix=''):
        assert key in self.custom_data
        import numpy as np
        mean_val = np.mean(self.custom_data[key])
        logger.record_tabular(prefix + key, mean_val)

    def clear_custom_data(self, key):
        if key in self.custom_data:
            del self.custom_data[key]
        else:
            logger.warn("[WARN] key [{}], not in custom_data".format(key))

    def get_custom_data(self, key):
        if key not in self.custom_data:
            return None
        else:
            return self.custom_data[key]

    def serialize_object_and_save(self):
        """
        This method is to save test object to a dill.
        This method will be call every time you call add_custom_record or other record function like self.check_and_test
        """
        # remove object which can is not serializable
        writer = self.writer
        self.writer = None
        saver = self.saver
        self.saver = None
        with open(self.pkl_file, 'wb') as f:
            dill.dump(self, f, recurse=True)
        self.writer = writer
        self.saver = saver

    def print_args(self):
        sort_list = sorted(self.hyper_param.items(), key=lambda i: i[0])
        for key, value in sort_list:
            # logger.info("key: %s, value: %s" % (key, value))
            logger.backup("key: %s, value: %s" % (key, value))
        # formatted_log_name = self.log_name_formatter(self.get_task_table_name(), self.record_date)
        params = exp_manager.hyper_param
        # params['formatted_log_name'] = formatted_log_name
        json.dump(params, open(osp.join(self.code_dir, 'parameter.json'), 'w'),
                  sort_keys=True, indent=4, allow_nan=True, default=lambda o: '<not serializable>')
        print("gen:", osp.join(self.code_dir, 'parameter.json'))


    def print_large_memory_variable(self):
        import sys
        large_mermory_dict = {}

        def sizeof_fmt(num, suffix='B'):
            for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
                if abs(num) < 1024.0:
                    return "%3.1f %s%s" % (num, unit, suffix), unit
                num /= 1024.0
            return "%.1f %s%s" % (num, 'Yi', suffix), 'Yi'

        for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                                 key=lambda x: -x[1])[:10]:
            size_str, fmt_type = sizeof_fmt(size)
            if fmt_type in ['', 'Ki', 'Mi']:
                continue
            logger.info("{:>30}: {:>8}".format(name, size_str))
            large_mermory_dict[str(name)] = size_str
        if large_mermory_dict != {}:
            summary = self.dict_to_table_text_summary(large_mermory_dict, 'large_memory')
            self.add_summary_to_logger(summary, 'large_memory')

    def dict_to_table_text_summary(self, input_dict, name):
        import tensorflow as tf
        with tf.Session(graph=tf.Graph()) as sess:
            to_tensor = [tf.convert_to_tensor([k, str(v)]) for k, v in input_dict.items()]
            return sess.run(tf.summary.text(name, tf.stack(to_tensor)))


exp_manager = tester = Tester()
