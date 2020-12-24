from RLA.easy_log.delete_log_tool import DeleteLogTool
import yaml
import argparse

def argsparser():
    parser = argparse.ArgumentParser("Delete Log")
    # reduce setting
    parser.add_argument('--sub_proj', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--reg', type=str)


    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = argsparser()
    dlt = DeleteLogTool(proj_root='./example/project_name/', sub_proj=args.sub_proj, task=args.task, regex=args.reg)
    dlt.delete_related_log()