
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

class DownloadLogTool(BasicLogTool):
    def __init__(self, rla_config_path, proj_root, sub_proj, task, regex, *args, **kwargs):
        fs = open(rla_config_path, encoding="UTF-8")
        self.private_config = yaml.load(fs)
        self.proj_root = proj_root
        self.sub_proj = sub_proj
        self.task = task
        self.regex = regex

    def _download_log(self, show=False):
        from RLA.auto_ftp import FTPHandler

        for log_type in self.log_types:
            root_dir_regex = osp.join(self.proj_root, self.sub_proj, log_type, self.task, self.regex)
            empty = True
            for root_dir in glob.glob(root_dir_regex):

                pass

class DeleteLogTool(BasicLogTool):
    def __init__(self, proj_root, sub_proj, task, regex, filter, *args, **kwargs):
        self.proj_root = proj_root
        self.sub_proj = sub_proj
        self.task = task
        self.regex = regex
        assert isinstance(filter, Filter)
        self.filter = filter
        self.small_timestep_regs = []
        super(DeleteLogTool, self).__init__(*args, **kwargs)

    def _find_small_timestep_log(self):
        root_dir_regex = osp.join(self.proj_root, self.sub_proj, LOG, self.task, self.regex)
        for root_dir in glob.glob(root_dir_regex):
            print("searching dirs", root_dir)
            if os.path.exists(root_dir):
                for file_list in os.walk(root_dir):

                    if re.search(r'\d{4}/\d{2}/\d{2}/\d{2}-\d{2}-\d{2}-\d{6}', file_list[0]):
                        target_reg = re.search(r'\d{4}/\d{2}/\d{2}/\d{2}-\d{2}-\d{2}-\d{6}', file_list[0]).group(0)
                    else:
                        target_reg = None
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
                                    self.small_timestep_regs.append(target_reg)
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
                                        if last_timestep < self.filter.timstep_bound:
                                            self.small_timestep_regs.append(target_reg)
                                            print("[delete] find an experiment with too small number of logs. ", file_list[0])
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
                                self.small_timestep_regs.append(target_reg)
                                print("[delete] find an experiment without any files. ", file_list[0])

    def _delete_related_log(self, regex, show=False):
        for log_type in self.log_types:
            root_dir_regex = osp.join(self.proj_root, self.sub_proj, log_type, self.task, regex)
            empty = True
            for root_dir in glob.glob(root_dir_regex):
                empty = False
                if os.path.exists(root_dir):
                    print("find a matched experiment", root_dir)
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
                        # if not show:
                        #     if len(os.listdir(file_list[0])) == 0:
                        #         cur_dir = file_list[0]
                        #         while True:
                        #             shutil.rmtree(cur_dir, ignore_errors=True)
                        #             print(" -- delete the empty dir", cur_dir, "---")
                        #             cur_dir = os.path.abspath(os.path.join(cur_dir, ".."))
                        #             if len(os.listdir(cur_dir)) != 0:
                        #                 break
                            # print("delete file {}".format(name))
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
                    print("not dir {}".format(root_dir))
            if empty: print("empty regex {}".format(root_dir_regex))

    def delete_related_log(self):
        self._delete_related_log(show=True, regex=self.regex)
        s = input("delete these files? (y/n)")
        if s == 'y':
            print("do delete ...")
            self._delete_related_log(show=False, regex=self.regex)

    def delete_small_timestep_log(self):
        self._find_small_timestep_log()
        print("complete searching.")
        s = input("show files to be deletes? (y/n)")
        if s == 'y':
            for reg in self.small_timestep_regs:
                print("[delete small-timestep log] reg: ", reg)
                self._delete_related_log(show=True, regex=reg + '*')
            s = input("delete these files? (y/n)")
            if s == 'y':
                for reg in self.small_timestep_regs:
                    print("do delete: ", reg)
                    self._delete_related_log(show=False, regex=reg + '*')


class ArchiveLogTool(BasicLogTool):
    def __init__(self, proj_root, sub_proj, task, regex, archive_name_as_task, remove, *args, **kwargs):
        self.proj_root = proj_root
        self.sub_proj = sub_proj
        self.task = task
        self.regex = regex
        self.remove = remove
        self.archive_name_as_task = archive_name_as_task
        super(ArchiveLogTool, self).__init__(*args, **kwargs)

    def _archive_log(self, show=False):
        for log_type in self.log_types:
            root_dir_regex = osp.join(self.proj_root, self.sub_proj, log_type, self.task, self.regex)
            archive_root_dir = osp.join(self.proj_root, self.sub_proj, log_type, self.archive_name_as_task)
            prefix_dir = osp.join(self.proj_root, self.sub_proj, log_type, self.task)
            prefix_len = len(prefix_dir)
            empty = True
            # os.system("chmod +x -R \"{}\"".format(prefix_dir))
            for root_dir in glob.glob(root_dir_regex):
                empty = False
                if os.path.exists(root_dir):
                    archiving_target = osp.join(archive_root_dir, root_dir[prefix_len+1:])
                    archiving_target_dir = '/'.join(archiving_target.split('/')[:-1])
                    os.makedirs(archiving_target_dir, exist_ok=True)
                    if os.path.isdir(root_dir):
                        if not show:
                            # os.makedirs(archiving_target, exist_ok=True)
                            shutil.copytree(root_dir, archiving_target)
                            if self.remove:
                                try:
                                    shutil.rmtree(root_dir)
                                except PermissionError as e:
                                    print("skip the permission error file")
                        print("move dir {}, to {}".format(root_dir, archiving_target))
                    else:
                        if not show:
                            shutil.copy(root_dir, archiving_target)
                            if self.remove:
                                os.remove(root_dir)
                        print("move file {}, to {}".format(root_dir, archiving_target))
                else:
                    print("not dir {}".format(root_dir))
            if empty: print("empty regex {}".format(root_dir_regex))
        pass

    def archive_log(self):
        self._archive_log(show=True)
        warn = ''
        if self.remove:
            warn = '[WARN] You are in the \'\'remove\'\' setting, the original log files will be removed!!'
        s = input("archive these files? (y/n) \n " + warn)
        if s == 'y':
            print("do archive ...")
            self._archive_log(show=False)

if __name__ == '__main__':
    dlt = DeleteLogTool("../", "var_seq_imitation", "self-transfer", "2019/11/29/01-11*")
    dlt.delete_related_log()
