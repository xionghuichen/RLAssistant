from RLA.easy_log.tester import tester
from RLA.easy_log import logger
from RLA.easy_log.tools import time_record, time_record_end
from RLA.easy_log.simple_mat_plot import simple_plot
import numpy as np
import argparse

def get_param():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Variational Sequence")
    parser.add_argument('--seed', help='RNG seed', type=int, default=88)
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v4')
    parser.add_argument('--load_date', default='')
    args = parser.parse_args()
    kwargs = vars(args)
    tester.set_hyper_param(**kwargs)
    tester.add_record_param(["env_id"])
    return kwargs

kwargs = get_param()
task_name = 'demo_task'
tester.configure(task_name, private_config_path='../../config.yaml', run_file='main.py')
tester.log_files_gen()
tester.print_args()

if kwargs["load_date"] is not '':
    from RLA.easy_log.tester import load_tester_from_record_date
    load_tester_from_record_date(fork_hp=False, task_name=task_name, record_date=kwargs["load_ddate"])
    start_epoch = tester.time_step_holder.get_time()
else:
    start_epoch = 0

import tensorflow as tf

X_ph = tf.placeholder(dtype=tf.float32, shape=[None, 32], name='x')
l = X_ph
for _ in range(10):
    l = tf.layers.dense(l, 16, kernel_initializer=tf.keras.initializers.glorot_normal)

sess = tf.Session().__enter__()
sess.run(tf.variables_initializer(tf.global_variables()))
tester.new_saver(var_prefix='', max_to_keep=1)
var_list = []
time_record('training')
for i in range(start_epoch, 1000):
    tester.time_step_holder.set_time(i)
    demo_log_var1 = np.random.randint(0, 1000)
    demo_log_var2 = i * np.random.randint(0, 2)
    var_list.append(demo_log_var2)
    logger.record_tabular("perf/var1", demo_log_var1)
    logger.record_tabular("perf/var2", demo_log_var2)
    logger.ma_record_tabular("ma/ma_var2", demo_log_var2, 10)
    if i % 50 == 0:
        tester.sync_log_file()
    if i % 100 == 0:
        tester.save_checkpoint()
    logger.dump_tabular()
simple_plot('var2', data=[var_list])  # the figure can be found in "results/experiment_name/var2.png"
time_record_end('training')

# show your log:

# 1. you can use ```tensorboard --logdir ./log/demo_task``` to check the log in tensorboard.
# 2. you can also use RLA.easy_plot.plot_func to plot the log in process.csv via jupyter notebook. for example:
from RLA.easy_plot.plot_func import plot_res_func
from RLA.easy_log.const import LOG
import datetime
import os
print(os.getcwd())
prefix_dir = './{}/{}'.format(LOG, task_name)
# filter the experiment name.
regex_of_your_log_date = str(tester.record_date.strftime("%Y/%m/%d/%H-%M")) + '*env_id=Hopper-v4*'
plot_res_func(prefix_dir, regs=[regex_of_your_log_date], split_keys=[], qualities=["ma/ma_var2", "perf/var2", "perf/var1"],
              smooth_step=5)
plot_res_func(prefix_dir, regs=[regex_of_your_log_date], split_keys=[], qualities=["ma/ma_var2", "perf/var2", "perf/var1"],
              smooth_step=5, replace_legend_keys=["A", "B", "C"], pretty=True, save_name='example.pdf')
# delete your log
from RLA.easy_log.delete_log_tool import DeleteLogTool
dlt = DeleteLogTool(proj_root='../', sub_proj='sub_task1', task=task_name, regex=regex_of_your_log_date)
dlt.delete_related_log()