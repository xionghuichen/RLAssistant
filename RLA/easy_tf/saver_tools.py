from RLA.easy_log im
# Saver manger.
def new_saver(self, var_prefix, max_to_keep):
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
    self.saver = tf.train.Saver(var_list=var_list, max_to_keep=max_to_keep, filename=self.checkpoint_dir,
                                save_relative_paths=True)


def save_checkpoint(self, iter=None):
    import tensorflow as tf
    if iter is None:
        iter = self.time_step_holder.get_time()
    self.saver.save(tf.get_default_session(), self.checkpoint_dir + 'checkpoint', global_step=iter)


def load_checkpoint(self):
    # TODO: load with variable scope.
    import tensorflow as tf
    logger.info("load checkpoint {}".format(self.checkpoint_dir))
    ckpt_path = tf.train.latest_checkpoint(self.checkpoint_dir)
    self.saver.restore(tf.get_default_session(), ckpt_path)
    max_iter = ckpt_path.split('-')[-1]
    self.time_step_holder.set_time(max_iter)
    return int(max_iter)