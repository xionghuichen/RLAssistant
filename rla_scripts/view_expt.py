"""
A script to view data of experiments.
"""

from RLA.easy_log.log_tools import ViewLogTool
import argparse
from config import *

def argsparser():
    parser = argparse.ArgumentParser("View Log")
    parser.add_argument('--task_table_name', type=str)
    parser.add_argument('--regex', type=str)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = argsparser()
    dlt = ViewLogTool(proj_root=DATA_ROOT, task_table_name=args.task_table_name, regex=args.regex)
    dlt.view_log()