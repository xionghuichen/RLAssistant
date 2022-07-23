
import numpy as np
import os.path as osp
import os
import shutil
import glob
import re
from RLA.easy_log.const import *
from RLA.const import DEFAULT_X_NAME
import stat
import distutils.dir_util
import yaml
import pandas as pd
import csv
import dill

import json
from RLA.easy_log.tester import Tester

class Filter(object):
    ALL = 'all'
    SMALL_TIMESTEP = 'small_ts'

    def config(self, type, timstep_bound):
        self.type = type
        self.timstep_bound = timstep_bound


class BasicLogTool(object):
    def __init__(self, optional_log_type=None):
        self.log_types = default_log_types.copy()
        if optional_log_type is not None:
            self.log_types.extend(optional_log_type)
    
    def is_valid_index(self, regex):
        if re.search(r'\d{4}/\d{2}/\d{2}/\d{2}-\d{2}-\d{2}-\d{6}', regex):
            target_reg = re.search(r'\d{4}/\d{2}/\d{2}/\d{2}-\d{2}-\d{2}-\d{6}', regex).group(0)
        else:
            target_reg = None
        return target_reg
    
    def _find_small_timestep_log(self, proj_root, task_table_name, regex, timstep_upper_bound=np.inf, timestep_lower_bound=0):
        small_timestep_regs = []
        root_dir_regex = osp.join(proj_root, LOG, task_table_name, regex)
        for root_dir in glob.glob(root_dir_regex):
            print("searching dirs", root_dir)
            if os.path.exists(root_dir):
                for file_list in os.walk(root_dir):
                    target_reg = self.is_valid_index(file_list[0])
                    if target_reg is not None:
                        if LOG in root_dir_regex:
                            try:
                                print(
                                    re.search(r'\d{4}/\d{2}/\d{2}/\d{2}-\d{2}-\d{2}-\d{6}', file_list[0]).group(1))
                                raise RuntimeError("found repeated timestamp")
                            except IndexError as e:
                                pass
                            progress_csv_file = file_list[0] + '/progress.csv'
                            if file_list[1] == ['tb'] or os.path.exists(progress_csv_file): # in root of logdir
                                if not os.path.exists(progress_csv_file) or os.path.getsize(progress_csv_file) == 0:
                                    print("[delete] find an experiment without progress.csv.", file_list[0])
                                    if timestep_lower_bound <= 0:
                                        small_timestep_regs.append([target_reg, file_list[0]])
                                else:
                                    try:
                                        reader = pd.read_csv(progress_csv_file, chunksize=100000, quoting=csv.QUOTE_NONE,
                                                             encoding='utf-8', index_col=False, comment='#')
                                        raw_df = pd.DataFrame()
                                        for chunk in reader:
                                            slim_chunk = chunk[[DEFAULT_X_NAME]]
                                            raw_df = pd.concat([raw_df, slim_chunk], ignore_index=True)
                                        last_timestep = raw_df[DEFAULT_X_NAME].max()
                                        print("[found a log] time_step ", last_timestep, target_reg)
                                        if timestep_lower_bound <= last_timestep <= timstep_upper_bound:
                                            small_timestep_regs.append([target_reg, file_list[0]])
                                            print("[delete] find an experiment satisfied timestep range. ", file_list[0])
                                        else:
                                            print("[valid]")
                                    except Exception as e:
                                        print("Load progress.csv failed", e, "reg", target_reg)
                                        pass
                            elif file_list[1] == ['events']: # in tb dir
                                pass
                            elif 'events' in file_list[0]: # in event dir
                                pass
                            else: # empty dir
                                if timestep_lower_bound <= 0:
                                    small_timestep_regs.append([target_reg, file_list[0]])
                                print("[delete] find an experiment without any files. ", file_list[0])
        return small_timestep_regs

class DownloadLogTool(BasicLogTool):
    def __init__(self, rla_config_path, proj_root, task, regex, *args, **kwargs):
        fs = open(rla_config_path, encoding="UTF-8")
        self.private_config = yaml.load(fs)
        self.proj_root = proj_root
        self.task_table_name = task
        self.regex = regex

    def _download_log(self, show=False):
        for log_type in self.log_types:
            root_dir_regex = osp.join(self.proj_root, log_type, self.task_table_name, self.regex)
            empty = True
            for root_dir in glob.glob(root_dir_regex):
                pass

class DeleteLogTool(BasicLogTool):
    def __init__(self, proj_root, task_table_name, regex, filter, *args, **kwargs):
        self.proj_root = proj_root
        self.task_table_name = task_table_name
        self.regex = regex
        assert isinstance(filter, Filter)
        self.filter = filter
        self.small_timestep_regs = []
        super(DeleteLogTool, self).__init__(*args, **kwargs)

    def _delete_related_log(self, regex, show=False, delete_log_types=None):
        log_found = 0
        for log_type in self.log_types:
            print(f"--- search {log_type} ---")
            if delete_log_types is not None and log_type not in delete_log_types:
                continue
            root_dir_regex = osp.join(self.proj_root, log_type, self.task_table_name, regex)
            empty = True
            for root_dir in glob.glob(root_dir_regex):
                empty = False
                if os.path.exists(root_dir):
                    print("find a matched experiment", root_dir)
                    log_found += 1
                    for file_list in os.walk(root_dir):
                        # walk into the leave of the file-tree.
                        for name in file_list[2]:
                            if not show:
                                # os.chmod(os.path.join(file_list[0], name), stat.S_IWRITE)
                                try:
                                    os.remove(os.path.join(file_list[0], name))
                                except PermissionError as e:
                                    print("skip the permission error file")
                        if not show:
                            print("delete sub-dir {}".format(file_list[0]))
                    if os.path.isdir(root_dir):
                        if not show:
                            try:
                                print("--- delete dir {} ---".format(root_dir))
                                shutil.rmtree(root_dir, ignore_errors=True)
                            except PermissionError as e:
                                print("skip the permission error file")
                    else:
                        if not show:
                            os.remove(root_dir)
                            print("--- delete root file {} ---".format(root_dir))
                else:
                    print("no dir {}".format(root_dir))
            if empty: print("empty regex {}".format(root_dir_regex))
        return log_found

    def delete_related_log(self, skip_ask=False, delete_log_types=None):
        self._delete_related_log(show=True, regex=self.regex, delete_log_types=delete_log_types)
        if skip_ask:
            s = 'y'
        else:
            s = input("delete these files? (y/n)")
        if s == 'y':
            print("do delete ...")
            return self._delete_related_log(show=False, regex=self.regex, delete_log_types=delete_log_types)
        else:
            return 0

    def delete_small_timestep_log(self, skip_ask=False):
        self.small_timestep_regs = self._find_small_timestep_log(self.proj_root, self.task_table_name, self.regex, timstep_upper_bound=self.filter.timstep_bound)
        print("complete searching.")
        if skip_ask:
            s = 'y'
        else:
            s = input("show files to be deleted? (y/n)")
        log_found = 0

        if s == 'y' or skip_ask:
            for res in self.small_timestep_regs:
                print("[delete small-timestep log] reg: ", res[1])
                self._delete_related_log(show=True, regex=res[0] + '*')
            print("summarize:")
            for count, res in enumerate(self.small_timestep_regs):
                print(f"[delete small-timestep log] {count} reg: {res[1]}")

            if skip_ask:
                s = 'y'
            else:
                s = input("delete these files? (y/n)")
            if s == 'y' or skip_ask:
                for res in self.small_timestep_regs:
                    print("do delete: ", res[1])
                    log_found += self._delete_related_log(show=False, regex=res[0] + '*')
        return log_found

class ArchiveLogTool(BasicLogTool):
    def __init__(self, proj_root, task_table_name, regex, archive_table_name=ARCHIVED_TABLE, *args, **kwargs):
        self.proj_root = proj_root
        self.task_table_name = task_table_name
        self.regex = regex
        self.archive_table_name = archive_table_name
        super(ArchiveLogTool, self).__init__(*args, **kwargs)

    def _archive_log(self, show=False):
        for log_type in self.log_types:
            root_dir_regex = osp.join(self.proj_root, log_type, self.task_table_name, self.regex)
            archive_root_dir = osp.join(self.proj_root, self.archive_table_name, log_type)
            prefix_dir = osp.join(self.proj_root, log_type)
            prefix_len = len(prefix_dir)
            empty = True
            # os.system("chmod +x -R \"{}\"".format(prefix_dir))
            for root_dir in glob.glob(root_dir_regex):
                empty = False
                if os.path.exists(root_dir):
                    # remove the overlapped path.
                    archiving_target = osp.join(archive_root_dir, root_dir[prefix_len+1:])
                    archiving_target_dir = '/'.join(archiving_target.split('/')[:-1])
                    os.makedirs(archiving_target_dir, exist_ok=True)
                    if os.path.isdir(root_dir):
                        if not show:
                            # os.makedirs(archiving_target, exist_ok=True)
                            shutil.copytree(root_dir, archiving_target)
                        print("copy dir {}, to {}".format(root_dir, archiving_target))
                    else:
                        if not show:
                            shutil.copy(root_dir, archiving_target)
                        print("copy file {}, to {}".format(root_dir, archiving_target))
                else:
                    print("no dir {}".format(root_dir))
            if empty: print("empty regex {}".format(root_dir_regex))
        pass

    def archive_log(self, skip_ask=False):
        self._archive_log(show=True)
        if skip_ask:
            s = 'y'
        else:
            s = input("archive these files? (y/n) \n ")
        if s == 'y':
            print("do archive ...")
            self._archive_log(show=False)


class ViewLogTool(BasicLogTool):
    def __init__(self, proj_root, task_table_name, regex, *args, **kwargs):
        self.proj_root = proj_root
        self.task_table_name = task_table_name
        self.regex = regex
        super(ViewLogTool, self).__init__(*args, **kwargs)

    def _view_log(self, regex):
        root_dir_regex = osp.join(self.proj_root, LOG, self.task_table_name, regex)
        for root_dir in glob.glob(root_dir_regex):
            if os.path.exists(root_dir):
                for file_list in os.walk(root_dir):
                    target_reg = self.is_valid_index(file_list[0])
                    if target_reg is not None:
                        backup_file = file_list[0] + '/backup.txt'
                        if file_list[1] == ['tb'] or os.path.exists(backup_file):  # in root of logdir
                            with open(backup_file) as f:
                                print(f.read())

    def view_log(self, skip_ask=False):
        found_regs = self._find_small_timestep_log(self.proj_root, self.task_table_name, self.regex, timestep_lower_bound=1)
        for res in found_regs:
            print("view experiments:", res[1])
            if skip_ask:
                s = 'y'
            else:
                s = input("press y to view \n ")
            if s == 'y':
                self._view_log(regex=res[0] + '*')


class PrettyPlotterTool(BasicLogTool):
    def __init__(self, proj_root, task_table_name, regex, *args, **kwargs):
        self.proj_root = proj_root
        self.task_table_name = task_table_name
        self.regex = regex
        super(PrettyPlotterTool, self).__init__(*args, **kwargs)

    def json_dump(self, location):
        target_index = self.is_valid_index(location)
        if target_index is not None:

            json_location = None
            try:
                exp_manager = dill.load(open(location, 'rb'))
                assert isinstance(exp_manager, Tester)
                formatted_log_name = exp_manager.log_name_formatter(exp_manager.get_task_table_name(),
                                                                    exp_manager.record_date)
                params = exp_manager.hyper_param
                params['formatted_log_name'] = formatted_log_name

                json_location = exp_manager.log_name_formatter(
                    osp.join(self.proj_root, LOG, exp_manager.get_task_table_name()), exp_manager.record_date) + '/'
                json.dump(params, open(osp.join(json_location, 'parameter.json'), 'w'))
                print("gen:", osp.join(json_location, 'parameter.json'))
            except FileNotFoundError as e:
                print("log file cannot found", json_location)
            except EOFError as e:
                print("log file broken", json_location)

    def gen_json(self, regex):
        root_dir_regex = osp.join(self.proj_root, ARCHIVE_TESTER, self.task_table_name, regex)
        for root_dir in glob.glob(root_dir_regex):
            if os.path.exists(root_dir):
                if osp.isdir(root_dir):
                    for file_list in os.walk(root_dir):
                        for file in file_list[2]:
                            location = osp.join(file_list[0], file)
                            self.json_dump(location)
                else:
                    self.json_dump(root_dir)
