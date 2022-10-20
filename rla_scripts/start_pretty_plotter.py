"""
A script to start a server of the pretty plotter.

"""
import os

from RLA.easy_log.log_tools import PrettyPlotterTool, Filter
import argparse
from RLA.rla_argparser import boolean_flag
from config import *

from smart_logger.front_page.page import start_page_server
import smart_logger.common.plot_config as plot_config
import smart_logger.common.page_config as page_config


def argsparser():
    parser = argparse.ArgumentParser("Delete Log")
    # reduce setting
    parser.add_argument('--task_table_name', type=str, default="")
    parser.add_argument('--regex', type=str)
    parser.add_argument('--timestep_bound', type=int, default=100)
    parser.add_argument('--delete_type', type=str, default=Filter.ALL)
    parser.add_argument('--workspace_path', '-wks', type=str, default='~/.pretty_plotter_cache',
                        help="Path to the workspace, used to saving cache data")
    parser.add_argument('--user_name', '-u', type=str, default='user',
                        help="user name")
    parser.add_argument('--password', '-pw', type=str, default='123456',
                        help="password")
    parser.add_argument('--port', '-p', type=int, default=7005, help="Server port")
    boolean_flag(parser, 'start_server', default=False)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = argsparser()
    filter = Filter()
    filter.config(type=args.delete_type, timstep_bound=args.timestep_bound)
    tool = PrettyPlotterTool(proj_root=DATA_ROOT, task_table_name=args.task_table_name, regex=args.regex)
    tool.gen_json(args.regex)
    if args.start_server:
        plot_config.DATA_PATH = os.path.abspath(DATA_ROOT)
        page_config.WORKSPAPCE = os.path.abspath(os.path.expanduser(args.workspace_path))

        plot_config.DATA_MERGER = []
        plot_config.PLOTTING_XY = []
        plot_config.PLOT_LOG_PATH = f"{plot_config.DATA_PATH}"
        plot_config.PLOT_FIGURE_SAVING_PATH = f"{os.path.join(os.path.dirname(plot_config.DATA_PATH), 'figure')}"

        page_config.WEB_RAM_PATH = f"{page_config.WORKSPAPCE}/WEB_ROM"
        page_config.CONFIG_PATH = f"{page_config.WEB_RAM_PATH}/configs"
        page_config.FIGURE_PATH = f"{page_config.WEB_RAM_PATH}/figures"
        page_config.PORT = args.port
        page_config.USER_NAME = args.user_name
        page_config.PASSWD = args.password
        start_page_server()
