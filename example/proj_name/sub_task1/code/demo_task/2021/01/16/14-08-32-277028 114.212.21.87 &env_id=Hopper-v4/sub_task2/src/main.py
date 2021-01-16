from RLA.easy_log.tester import tester
from RLA.easy_log import logger
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

tester.new_saver(var_prefix='', max_to_keep=1)
for i in range(start_epoch, 1000):
    tester.time_step_holder.set_time(i)
    demo_log_var1 = np.random.randint(0, 100)
    demo_log_var2 = i * np.random.randint(0, 2)
    logger.record_tabular("perf/var1", demo_log_var1)
    logger.record_tabular("perf/var2", demo_log_var2)
    logger.ma_record_tabular("ma/ma_var2", demo_log_var2, 10)
    if i % 50 == 0:
        tester.sync_log_file()
    if i % 100 == 0:
        tester.save_checkpoint()

