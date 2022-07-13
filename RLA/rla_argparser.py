import argparse


def boolean_flag(parser: argparse.ArgumentParser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def arg_parser_postprocess(parser: argparse.ArgumentParser):
    parser.add_argument('--loaded_task_name', default='', type=str)
    parser.add_argument('--info', default='default exp info', type=str)
    parser.add_argument('--loaded_date', default=True, type=str)
    return parser
