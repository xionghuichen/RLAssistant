# Created by xionghuichen at 2022/7/29
# Email: chenxh@lamda.nju.edu.cn
from RLA.easy_log import logger
import time


rc_start_time = {}


def time_record(name):
    """
    record the consumed time of your code snippet. call this function to start a recorder.
    "name" is identifier to distinguish different recorder and record different snippets at the same time.
    call time_record_end to end a recorder.
    :param name: identifier of your code snippet.
    :type name: str
    :return:
    :rtype:
    """
    assert name not in rc_start_time
    rc_start_time[name] = time.time()


def time_record_end(name):
    """
    record the consumed time of your code snippet. call this function to start a recorder.
    "name" is identifier to distinguish different recorder and record different snippets at the same time.
    call time_record_end to end a recorder.
    :param name: identifier of your code snippet.
    :type name: str
    :return:
    :rtype:
    """
    end_time = time.time()
    start_time = rc_start_time[name]
    logger.record_tabular("time_used/{}".format(name), end_time - start_time)
    logger.info("[test] func {0} time used {1:.2f}".format(name, end_time - start_time))
    del rc_start_time[name]