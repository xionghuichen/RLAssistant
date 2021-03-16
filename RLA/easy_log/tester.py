#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017-11-12
# Modified    :   2017-11-12
# Version     :   1.0
from collections import deque
import dill
import time
import os

import datetime
import os.path as osp
from RLA.easy_log.const import *
from RLA.easy_log.time_step import time_step_holder
from RLA.easy_log import logger
from RLA.easy_log.const import *
import yaml
import shutil


def load_from_record_date(task_name, record_date, fork_hp):
    global tester
    assert isinstance(tester, Tester)
    load_tester = tester.load_tester(record_date, task_name, tester.root)
    # copy log file
    tester.log_file_copy(load_tester)
    # copy attribute
    if fork_hp:
        tester.hyper_param = load_tester.hyper_param
        tester.hyper_param_record = load_tester.hyper_param_record
        tester.private_config = load_tester.private_config
    # load checkpoint
    load_tester.new_saver(var_prefix='', max_to_keep=1)
    load_iter, load_res = load_tester.load_checkpoint()
    tester.time_step_holder.set_time(load_iter)
    tester.print_log_dir()
    return load_iter, load_res


class Tester(object):

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

    def configure(self, task_name, private_config_path, run_file, log_root):
        """

        :param task_name:
        :param private_config_path:
        :return:
        """
        fs = open(private_config_path, encoding="UTF-8")
        self.private_config = yaml.load(fs)
        self.run_file = run_file
        self.task_name = task_name
        self.root = log_root
        logger.info("private_config: ")
        self.dl_framework = self.private_config["DL_FRAMEWORK"]
        self.project_root = "/".join(private_config_path.split("/")[:-1])
        for k, v in self.private_config.items():
            logger.info("k: {}, v: {}".format(k, v))

    def set_hyper_param(self, **argkw):
        """
        This method is to record all of hyper parameters to test object.

        Place pass your parameters as follow format:
            self.set_hyper_param(param_a=a,param_b=b)

        Note: It is invalid to pass a local object to this function.

        Parameters
        ----------
        argkw : key-value
            for example: self.set_hyper_param(param_a=a,param_b=b)

        """
        self.hyper_param = argkw

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
        code_dir, _ = self.__create_file_directory(osp.join(self.root, CODE, self.task_name), '', is_file=False)
        log_dir, _ = self.__create_file_directory(osp.join(self.root, LOG, self.task_name), '', is_file=False)
        self.pkl_dir, self.pkl_file = self.__create_file_directory(osp.join(self.root, ARCHIVE_TESTER, self.task_name), '.pkl')
        self.checkpoint_dir, _ = self.__create_file_directory(osp.join(self.root, CHECKPOINT, self.task_name), is_file=False)
        self.results_dir, _ = self.__create_file_directory(osp.join(self.root, OTHER_RESULTS, self.task_name), is_file=False)
        self.log_dir = log_dir
        self.code_dir = code_dir

        self.init_logger()
        self.serialize_object_and_save()
        self.__copy_source_code(self.run_file, code_dir)
        self.feed_hyper_params_to_tb()
        self.print_log_dir()

    def update_log_files_location(self, root):
        self.root = root
        code_dir, _ = self.__create_file_directory(osp.join(self.root, CODE, self.task_name), '', is_file=False)
        log_dir, _ = self.__create_file_directory(osp.join(self.root, LOG, self.task_name), '', is_file=False)
        self.pkl_dir, self.pkl_file = self.__create_file_directory(osp.join(self.root, ARCHIVE_TESTER, self.task_name), '.pkl')
        self.checkpoint_dir, _ = self.__create_file_directory(osp.join(self.root, CHECKPOINT, self.task_name), is_file=False)
        self.results_dir, _ = self.__create_file_directory(osp.join(self.root, OTHER_RESULTS, self.task_name), is_file=False)
        self.log_dir = log_dir
        self.code_dir = code_dir
        self.print_log_dir()

    def init_logger(self):
        self.writer = None
        # logger configure
        logger.info("store file %s" % self.pkl_file)
        logger.configure(self.log_dir, self.private_config["LOG_USED"])
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
        self.init_logger()

    def task_gen(self, task_pattern_list):
        return '-'.join(task_pattern_list)

    def print_log_dir(self):
        logger.info("log dir: {}".format(self.log_dir))
        logger.info("pkl_file: {}".format(self.pkl_file))
        logger.info("checkpoint_dir: {}".format(self.checkpoint_dir))
        logger.info("results_dir: {}".format(self.results_dir))

    @classmethod
    def load_tester(cls, record_date, task_name, log_root):
        logger.info("load tester")
        res_dir, res_file = cls.log_file_finder(record_date, task_name=task_name,
                                                file_root=osp.join(log_root, ARCHIVE_TESTER),
                                                log_type='files')
        import dill
        load_tester = dill.load(open(res_dir+'/'+res_file, 'rb'))
        assert isinstance(load_tester, Tester)
        logger.info("update log files' root")
        load_tester.update_log_files_location(root=log_root)
        return load_tester


    def add_record_param(self, keys):
        for k in keys:
            if '.' in k:
                try:
                    sub_k_list = k.split('.')
                    v = self.hyper_param[sub_k_list[0]]
                    for sub_k in sub_k_list[1:]:
                        v = v[sub_k]
                    self.hyper_param_record.append(str(k) + '=' + str(v).replace('[', '{').replace(']', '}'))
                except KeyError as e:
                    print("do not include dot ('.') in your hyperparemeter name")
            else:
                self.hyper_param_record.append(str(k) + '=' + str(self.hyper_param[k]).replace('[', '{').replace(']', '}'))

    def add_summary_to_logger(self, summary, name='', simple_val=False, freq=20):
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

    def feed_hyper_params_to_tb(self):
        if "tensorboard" not in self.private_config["LOG_USED"]:
            logger.info("skip feeding hyper-param to tb")
            return

        import tensorflow as tf
        with tf.Session(graph=tf.Graph()) as sess:
            hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in self.hyper_param.items()]
            summary = sess.run(tf.summary.text('hyperparameters', tf.stack(hyperparameters)))
        self.add_summary_to_logger(summary, 'hyperparameters', freq=1)

    def sync_log_file(self):
        """
        syn_log_file is an automatic synchronization function.
        It will send all log files (e.g., code/**, checkpoint/**, log/**, etc.) to your target server via the FTP protocol.
        To run this function, you should add some configuration on SRG.private_config.py

        SEND_LOG_FILE: boolean. denotes synchronization or not.
        ftp_server: target server ip address
        username: username of target server
        password: password of target server
        remote_porject_dir: log root of target server, e.g., "/Project/SRG/SRG/var_gan_imitation/"

        :return:
        """

        logger.warn("sync: start")
        # ignore_files = self.private_config["IGNORE_RULE"]
        if self.private_config["SEND_LOG_FILE"]:
            from RLA.auto_ftp import FTPHandler
            try:
                ftp = FTPHandler(ftp_server=self.private_config["REMOTE_SETTING"]["ftp_server"],
                                 username=self.private_config["REMOTE_SETTING"]["username"],
                                 password=self.private_config["REMOTE_SETTING"]["password"])
                for root, dirs, files in os.walk(self.log_dir):
                    suffix = root.split("/{}/".format(LOG))
                    assert len(suffix) == 2, "root should have only one pattern \"/log/\""
                    remote_root = osp.join(self.private_config["REMOTE_SETTING"]["remote_log_root"], LOG, suffix[1])
                    local_root = root
                    logger.warn("sync {} <- {}".format(remote_root, local_root))
                    for file in files:
                        ftp.upload_file(remote_root, local_root, file)
                # for root, dirs, files in os.walk(self.code_dir):
                #     remote_root = osp.join(self.private_config.remote_porject_dir, root[3:])
                #     local_root = root
                #     logger.warn("sync {} <- {}".format(remote_root, local_root))
                #     for file in files:
                #         ftp.upload_file(remote_root, local_root, file)
                # for root, dirs, files in os.walk(self.checkpoint_dir):
                #     for file in files:
                #         ftp.upload_file(remote_porject_dir + root[2:], root + '/', file)

                logger.warn("sync: send success!")
            except Exception as e:
                logger.warn("sending log file failed. {}".format(e))
                import traceback
                logger.warn(traceback.format_exc())

    @classmethod
    def log_file_finder(cls, record_date, task_name='train', file_root='../checkpoint/', log_type='dir'):
        record_date = datetime.datetime.strptime(record_date, '%Y/%m/%d/%H-%M-%S-%f')
        prefix = osp.join(file_root, task_name)
        directory = str(record_date.strftime("%Y/%m/%d"))
        directory = osp.join(prefix, directory)
        file_found = ''
        for root, dirs, files in os.walk(directory):
            if log_type == 'dir':
                search_list = dirs
            elif log_type =='files':
                search_list =files
            else:
                raise NotImplementedError
            for search_item in search_list:
                if search_item.startswith(str(record_date.strftime("%H-%M-%S-%f"))):
                    split_dir = search_item.split(' ')
                    # self.__ipaddr = split_dir[1]
                    info = " ".join(split_dir[2:])
                    logger.info("load data: \n ts {}, \n ip {}, \n info {}".format(split_dir[0], split_dir[1], info))
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

    def __copy_source_code(self, run_file, code_dir):
        import shutil
        if self.private_config["PROJECT_TYPE"]["backup_code_by"] == 'lib':
            assert os.listdir(code_dir) == []
            os.removedirs(code_dir)
            shutil.copytree(osp.join(self.project_root, self.private_config["BACKUP_CONFIG"]["lib_dir"]), code_dir)
            shutil.copy(run_file, code_dir)
        elif self.private_config["PROJECT_TYPE"]["backup_code_by"] == 'source':
            for dir_name in self.private_config["BACKUP_CONFIG"]["backup_code_dir"]:
                shutil.copytree(osp.join(self.project_root, dir_name), code_dir + '/' + dir_name)
        else:
            raise NotImplementedError

    def record_date_to_str(self, record_date):
        return str(record_date.strftime("%H-%M-%S-%f"))

    def __create_file_directory(self, prefix, ext='', is_file=True, record_date=None):
        if record_date is None:
            record_date = self.record_date
        directory = str(record_date.strftime("%Y/%m/%d"))
        directory = osp.join(prefix, directory)
        if is_file:
            os.makedirs(directory, exist_ok=True)
            file_name = '{dir}/{timestep} {ip} {info}{ext}'.format(dir=directory,
                                                                 timestep=self.record_date_to_str(record_date),
                                                                 ip=str(self.ipaddr),
                                                                 info=self.info,
                                                                 ext=ext)
        else:
            directory = '{dir}/{timestep} {ip} {info}{ext}/'.format(dir=directory,
                                                                 timestep=self.record_date_to_str(record_date),
                                                                 ip=str(self.ipaddr),
                                                                 info=self.info,
                                                                 ext=ext)
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

    def time_record(self, name):
        assert name not in self._rc_start_time
        self._rc_start_time[name] = time.time()

    def time_record_end(self, name):
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
        if self.dl_framework == 'tensorflow':
            import tensorflow as tf
            if var_prefix is None:
                var_prefix = ''
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, var_prefix)
            logger.info("save variable :")
            for v in var_list:
                logger.info(v)
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=max_to_keep, filename=self.checkpoint_dir, save_relative_paths=True)
        elif self.dl_framework == 'pytorch':
            self.max_to_keep = max_to_keep
            self.checkpoint_keep_list = []
        else:
            raise NotImplementedError

    def save_checkpoint(self, model_dict=None):
        if self.dl_framework == 'tensorflow':
            import tensorflow as tf
            iter = self.time_step_holder.get_time()
            cpt_name = osp.join(self.checkpoint_dir, 'checkpoint')
            logger.info("save checkpoint to ", cpt_name, iter)
            self.saver.save(tf.get_default_session(), cpt_name, global_step=iter)
        elif self.dl_framework == 'pytorch':
            import torch
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

    def load_checkpoint(self):
        if self.dl_framework == 'tensorflow':
            # TODO: load with variable scope.
            import tensorflow as tf
            cpt_name = osp.join(self.checkpoint_dir)
            logger.info("load checkpoint {}".format(cpt_name))
            ckpt_path = tf.train.latest_checkpoint(cpt_name)
            self.saver.restore(tf.get_default_session(), ckpt_path)
            max_iter = ckpt_path.split('-')[-1]
            self.time_step_holder.set_time(max_iter)
            return int(max_iter), None
        elif self.dl_framework == 'pytorch':
            import torch
            return self.checkpoint_keep_list[-1], torch.load(tester.checkpoint_dir + "checkpoint-{}.pt".format(self.checkpoint_keep_list[-1]))

    def auto_parse_info(self):
        return '&'.join(self.hyper_param_record)


    def add_graph(self, sess):
        assert self.writer is not None
        self.writer.add_graph(sess.graph)

    # --- custom data manager --
    def add_custom_data(self, key, data, dtype=list, max_len=-1):
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
            dill.dump(self, f)
        self.writer = writer
        self.saver = saver

    def print_args(self):
        sort_list = sorted(self.hyper_param.items(), key=lambda i: i[0])
        for key, value in sort_list:
            logger.info("key: %s, value: %s" % (key, value))

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


tester = Tester()
