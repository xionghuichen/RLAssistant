"""
A script to archive benchmarking experiments. 

It is convenient to merge the archived experiments and the current task into tensorboard by:

tensorboard --logdir ./log/your_task/,./log/archived/

"""

from RLA.easy_log.log_tools import ArchiveLogTool
import argparse
from config import *

def argsparser():
    parser = argparse.ArgumentParser("Archive Log")
    # reduce setting
    parser.add_argument('--task_table_name', type=str)
    parser.add_argument('--archive_table_name', type=str, default=ARCHIVED_TABLE)
    parser.add_argument('--regex', type=str)
    parser.add_argument('--remove', action='store_true')


    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = argsparser()
    dlt = ArchiveLogTool(proj_root=DATA_ROOT, task_table_name=args.task_table_name, regex=args.regex,
                         archive_table_name=args.archive_table_name, remove=args.remove)
    dlt.archive_log()