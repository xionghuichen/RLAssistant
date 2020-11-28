from common.delete_log_tool import DeleteLogTool
from common.private_config import *
import argparse

def argsparser():
    parser = argparse.ArgumentParser("Delete Log")
    parser.add_argument('--log_root', type=str, default=LOG_ROOT)
    # reduce setting
    parser.add_argument('--sub_proj', type=str, default=DEFAULT_PROJ)
    parser.add_argument('--task', type=str, default=DEFAULT_TASK)
    parser.add_argument('--reg', type=str, default='')


    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = argsparser()
    dlt = DeleteLogTool(args.log_root, args.sub_proj, args.task, args.reg)
    dlt.delete_related_log()
