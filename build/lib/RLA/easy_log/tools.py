from RLA.easy_log import logger
import time
import os.path as osp
import os
import shutil
import glob
from RLA.easy_log.const import *
import stat

class DeleteLogTool(object):
    def __init__(self, log_root, sub_proj, task, regex, config, optional_log_type=None):
        self.log_root = log_root
        self.sub_proj = sub_proj
        self.task = task
        self.config = config
        self.regex = regex
        self.log_types = default_log_types.copy()
        if optional_log_type is not None:
            self.log_types.extend(optional_log_type)

    def _delete_related_log(self, show=False):
        for log_type in self.log_types:
            root_dir_regex = osp.join(self.log_root, self.sub_proj, log_type, self.task, self.regex)
            empty = True
            for root_dir in glob.glob(root_dir_regex):
                empty = False
                if os.path.exists(root_dir):
                    for file_list in os.walk(root_dir): # walk into the leave of the file-tree.
                        for name in file_list[2]:
                            if not show:
                                os.chmod(os.path.join(file_list[0], name), stat.S_IWRITE)
                                os.remove(os.path.join(file_list[0], name))
                            # print("delete file {}".format(name))
                    if os.path.isdir(root_dir):
                        if not show:
                            shutil.rmtree(root_dir)
                        print("delete dir {}".format(root_dir))
                    else:
                        if not show:
                            os.remove(root_dir)
                        print("delete file {}".format(root_dir))
                else:
                    print("not dir {}".format(root_dir))
            if empty: print("empty regex {}".format(root_dir_regex))

    def delete_related_log(self):
        self._delete_related_log(show=True)
        s = input("delete these files? (y/n)")
        if s == 'y':
            print("do delete ...")
            self._delete_related_log(show=False)


def time_used_wrap(name, func, *args, **kwargs):
    start_time = time.time()
    output = func(*args, **kwargs)
    end_time = time.time()
    time_used = end_time - start_time
    logger.info("[test] func {0} time used {1:.2f}".format(name, time_used))
    logger.record_tabular("time_used/{}".format(name), time_used)
    logger.dump_tabular()
    return output


rc_start_time = {}

def time_record(name):
    assert name not in rc_start_time
    rc_start_time[name] = time.time()

def time_record_end(name):
    end_time = time.time()
    start_time = rc_start_time[name]
    logger.record_tabular("time_used/{}".format(name), end_time - start_time)
    logger.info("[test] func {0} time used {1:.2f}".format(name, end_time - start_time))
    del rc_start_time[name]