import argparse


def arg_parser_postprocess(parser: argparse.ArgumentParser):
    parser.add_argument('--loaded_task_name', default='', type=str)
    parser.add_argument('--info', default='default exp info', type=str)
    parser.add_argument('--loaded_date', default=True, type=str)
    return parser
