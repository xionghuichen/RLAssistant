
import numpy as np
import os.path as osp
import os
import shutil
import glob
from RLA.easy_log.const import *
import stat
import distutils.dir_util

class BasicLogTool(object):
    def __init__(self, optional_log_type=None):
        self.log_types = default_log_types.copy()
        if optional_log_type is not None:
            self.log_types.extend(optional_log_type)


class DeleteLogTool(BasicLogTool):
    def __init__(self, proj_root, sub_proj, task, regex, *args, **kwargs):
        self.proj_root = proj_root
        self.sub_proj = sub_proj
        self.task = task
        self.regex = regex
        super(DeleteLogTool, self).__init__(*args, **kwargs)

    def _delete_related_log(self, show=False):
        for log_type in self.log_types:
            root_dir_regex = osp.join(self.proj_root, self.sub_proj, log_type, self.task, self.regex)
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
                                shutil.rmtree(root_dir)
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
