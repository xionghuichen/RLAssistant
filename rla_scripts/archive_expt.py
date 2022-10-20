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
    parser.add_argument('--regex', type=str)

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = argsparser()
    dlt = ArchiveLogTool(proj_root=DATA_ROOT, task_table_name=args.task_table_name, regex=args.regex)
    dlt.archive_log()