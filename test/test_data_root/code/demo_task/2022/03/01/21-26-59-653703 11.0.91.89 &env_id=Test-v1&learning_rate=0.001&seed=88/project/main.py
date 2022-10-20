from RLA.easy_log.tester import exp_manager
from RLA.easy_log import logger
from RLA.easy_log.tools import time_record, time_record_end
from RLA.easy_log.simple_mat_plot import simple_plot
from RLA.rla_argparser import arg_parser_postprocess
from RLA.easy_log.complex_data_recorder import MatplotlibRecorder as mpr
import numpy as np
import tensorflow as tf
import argparse

# phase 1: init your hyper-parameters
def get_param():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Variational Sequence")
    parser.add_argument('--seed', help='RNG seed', type=int, default=88)
    parser.add_argument('--env_id', help='environment ID', default='Test-v1')
    parser.add_argument('--learning_rate', help='a hyperparameter', default=1e-3, type=float)
    parser.add_argument('--input_size', help='a hyperparameter', default=16, type=int)
    parser = arg_parser_postprocess(parser)
    args = parser.parse_args()
    kwargs = vars(args)
    exp_manager.set_hyper_param(**kwargs)
    exp_manager.add_record_param(["env_id", "learning_rate", "seed"])
    return kwargs

# phase 2: init the RLA experiment manager.
kwargs = get_param()

task_name = 'demo_task'
rla_data_root = '../'
exp_manager.configure(task_name, private_config_path='../rla_config.yaml', data_root=rla_data_root)
exp_manager.log_files_gen()
exp_manager.print_args()

def set_global_seeds(seed):
    """
    set the seed for python random, tensorflow, numpy and gym spaces

    :param seed: (int) the seed
    """
    import random
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_global_seeds(kwargs["seed"])

# phase 3: [optional] resume from a historical experiment.
from RLA.easy_log.exp_loader import exp_loader
exp_loader.fork_log_files()
start_epoch, load_res, hist_variables = exp_loader.load_from_record_date(variable_list=['iv'])

# phase 4: write your code.
X_ph = tf.placeholder(dtype=tf.float32, shape=[None, kwargs["input_size"]], name='x')
y_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')
l = X_ph
for _ in range(3):
    l = tf.nn.tanh(tf.layers.dense(l, 64, kernel_initializer=tf.keras.initializers.glorot_normal))

out = tf.layers.dense(l, 1, kernel_initializer=tf.keras.initializers.glorot_normal)
loss = tf.reduce_mean(np.square(out - y_ph))
opt = tf.train.AdamOptimizer(learning_rate=kwargs["learning_rate"]).minimize(loss)

sess = tf.Session().__enter__()
sess.run(tf.variables_initializer(tf.global_variables()))

exp_manager.new_saver(var_prefix='', max_to_keep=1)


def target_func(x):
    return np.tanh(np.mean(x, axis=-1, keepdims=True))


for i in range(start_epoch, 1000):
    exp_manager.time_step_holder.set_time(i)
    x_input = np.random.normal(0, 3, [64, kwargs["input_size"]])
    y = target_func(x_input)
    loss_out, y_pred = sess.run([loss, out, opt], feed_dict={X_ph:x_input, y_ph: y})[:-1]
    logger.ma_record_tabular("perf/mse", loss_out, 10)
    logger.record_tabular("y_out", np.mean(y))
    logger.dump_tabular()
    if i % 100 == 0:
        exp_manager.save_checkpoint()
    if i % 10 == 0:
        def plot_func():
            import matplotlib.pyplot as plt
            testX = np.repeat(np.expand_dims(np.arange(-10, 10, 0.1), axis=-1), repeats=16, axis=-1)
            testY = target_func(testX)
            predY = sess.run(out, feed_dict={X_ph: testX})
            plt.plot(testX.mean(axis=-1), predY.mean(axis=-1), label='pred')
            plt.plot(testX.mean(axis=-1), testY.mean(axis=-1), label='real')
        mpr.pretty_plot_wrapper('react_func', plot_func, xlabel='x', ylabel='y', title='react test')
    logger.dump_tabular()