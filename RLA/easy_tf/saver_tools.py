from RLA.easy_log import logger
from RLA.easy_log.tester import tester
from RLA.easy_log.const import *


def load_tester_from_record_date(task_name, record_date, fork_hp):
    global tester
    load_tester = tester.load_tester(record_date, task_name, tester.root + ARCHIVE_TESTER + '/')
    # copy log file
    tester.log_file_copy(load_tester)
    # copy attribute
    if fork_hp:
        tester.hyper_param = load_tester.hyper_param
        tester.hyper_param_record = load_tester.hyper_param_record
        tester.private_config = load_tester.private_config
    # load checkpoint
    saver = new_saver(var_prefix='', max_to_keep=1, checkpoint_path=load_tester)
    max_iter = load_checkpoint(saver=saver, tester=load_tester)
    tester.time_step_holder.set_time(max_iter)
    tester.print_log_dir()


# Saver manger.
def new_saver(max_to_keep, var_prefix, checkpoint_path=None):
    """
    initialize new tf.Saver
    :param var_prefix: we use var_prefix to filter the variables for saving.
    :param max_to_keep:
    :return:
    """
    import tensorflow as tf
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, var_prefix)
    logger.info("save variable :")
    for v in var_list:
        logger.info(v)
    if checkpoint_path:
        checkpoint_path = tester.checkpoint_path
    saver = tf.train.Saver(var_list=var_list, max_to_keep=max_to_keep, filename=checkpoint_path, save_relative_paths=True)
    return saver


def save_checkpoint(saver, iter=None):
    import tensorflow as tf
    if iter is None:
        iter = tester.time_step_holder.get_time()
    saver.save(tf.get_default_session(), tester.checkpoint_dir + 'checkpoint', global_step=iter)


def load_checkpoint(saver, tester):
    # TODO: load with variable scope.
    import tensorflow as tf
    logger.info("load checkpoint {}".format(tester.checkpoint_dir))
    ckpt_path = tf.train.latest_checkpoint(tester.checkpoint_dir)
    saver.restore(tf.get_default_session(), ckpt_path)
    max_iter = ckpt_path.split('-')[-1]
    tester.time_step_holder.set_time(max_iter)
    return int(max_iter)