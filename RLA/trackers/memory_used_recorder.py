# Created by xionghuichen at 2022/11/25
# Email: chenxh@lamda.nju.edu.cn
import sys


from RLA import logger, exp_manager


def print_large_memory_variable():
    large_mermory_dict = {}

    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix), unit
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix), 'Yi'

    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key=lambda x: -x[1])[:10]:
        size_str, fmt_type = sizeof_fmt(size)
        if fmt_type in ['', 'Ki', 'Mi']:
            continue
        logger.info("{:>30}: {:>8}".format(name, size_str))
        large_mermory_dict[str(name)] = size_str
    if large_mermory_dict != {}:
        summary = exp_manager.dict_to_table_text_summary(large_mermory_dict, 'large_memory')
        exp_manager.add_summary_to_logger(summary, 'large_memory')