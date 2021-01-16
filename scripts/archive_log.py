from RLA.easy_log.delete_log_tool import ArchiveLogTool
import argparse

def argsparser():
    parser = argparse.ArgumentParser("Archive Log")
    # reduce setting
    parser.add_argument('--sub_proj', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--archive_name_as_task', type=str, default='archived')
    parser.add_argument('--reg', type=str)
    parser.add_argument('--remove', action='store_true')


    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = argsparser()
    dlt = ArchiveLogTool(proj_root='./example/project_name/', sub_proj=args.sub_proj, task=args.task, regex=args.reg,
                         archive_name_as_task=args.archive_name_as_task, remove=args.remove)
    dlt.archive_log()