import numpy as np
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

if __name__ == '__main__':
    dlt = DeleteLogTool("../", "var_seq_imitation", "self-transfer", "2019/11/29/01-11*")
    dlt.delete_related_log()
