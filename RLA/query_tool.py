# Created by xionghuichen at 2022/8/10
# Email: chenxh@lamda.nju.edu.cn

import glob
import os.path as osp
import os
import dill
import re
import copy
from RLA.easy_log.const import LOG, ARCHIVE_TESTER, LogDataType
from RLA.easy_log.tester import Tester


class BasicQueryResult(object):
    def __init__(self, dirname):
        self.dirname = dirname

class ArchiveQueryResult(BasicQueryResult):
    def __init__(self, exp_manager, dirname):
        super(ArchiveQueryResult, self).__init__(dirname)
        assert isinstance(exp_manager, Tester)
        self.exp_manager = exp_manager

class LogQueryResult(BasicQueryResult):
    def __init__(self, dirname):
        super(LogQueryResult, self).__init__(dirname)

def extract_valid_index(regex):
    if re.search(r'\d{4}/\d{2}/\d{2}/\d{2}-\d{2}-\d{2}-\d{6}', regex):
        target_reg = re.search(r'\d{4}/\d{2}/\d{2}/\d{2}-\d{2}-\d{2}-\d{6}', regex).group(0)
    else:
        target_reg = None
    return target_reg

def experiment_data_query(data_root, task_table_name, reg, data_type):
    if data_type == LOG:
        return _log_data_query(data_root, task_table_name, reg)
    elif data_type == ARCHIVE_TESTER:
        return _archive_tester_query(data_root, task_table_name, reg)
    else:
        raise NotImplementedError


def _archive_tester_query(data_root, task_table_name, reg):
    experiment_data_dict = {}
    root_dir_regex = osp.join(data_root, ARCHIVE_TESTER, task_table_name, reg)
    for root_dir in glob.glob(root_dir_regex):
        if os.path.exists(root_dir):
            if osp.isdir(root_dir):
                for file_list in os.walk(root_dir):
                    for file in file_list[2]:
                        location = osp.join(file_list[0], file)
                        exp_manager = dill.load(open(location, 'rb'))
                        dirname = location.split('.pkl')[0]
                        key = extract_valid_index(location)
                        experiment_data_dict[key] = ArchiveQueryResult(dirname=dirname, exp_manager=exp_manager)
            else:
                location = root_dir
                key = extract_valid_index(location)
                exp_manager = dill.load(open(location, 'rb'))
                dirname = location.split('.pkl')[0]
                experiment_data_dict[key] = ArchiveQueryResult(dirname=dirname, exp_manager=exp_manager)
    return experiment_data_dict


def _log_data_query(data_root, task_table_name, reg):
    experiment_data_dict = {}
    root_dir_regex = osp.join(data_root, LOG, task_table_name, reg)
    for root_dir in glob.glob(root_dir_regex):
        if os.path.exists(root_dir):
            if osp.isdir(root_dir):
                for file_list in os.walk(root_dir):
                    for file in file_list[2]:
                        if 'progress.csv' in file:
                            location = osp.join(file_list[0], file)
                            key = extract_valid_index(location)
                            dirname = osp.dirname(location)
                            experiment_data_dict[key] = LogQueryResult(dirname=dirname)
                            break
            else:
                location = root_dir
                dirname = osp.dirname(location)
                experiment_data_dict[key] = LogQueryResult(dirname=dirname)
    return experiment_data_dict

