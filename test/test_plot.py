# Created by xionghuichen at 2022/8/10
# Email: chenxh@lamda.nju.edu.cn
from test._base import BaseTest
from RLA.easy_log.log_tools import DeleteLogTool, Filter
from RLA.easy_log.log_tools import ArchiveLogTool, ViewLogTool
from RLA.easy_log.tester import exp_manager

import os

class ScriptTest(BaseTest):
    def test_plot(self):
        from RLA import plot_func
        data_root = 'test_data_root'
        task = 'demo_task'
        regs = [
            '2022/03/01/21-[12]*'
        ]
        _ = plot_func(data_root=data_root, task_table_name=task, regs=regs, split_keys=['learning_rate'],
                      metrics=['perf/mse'])
        _ = plot_func(data_root=data_root, task_table_name=task, regs=regs, split_keys=['learning_rate'],
                      metrics=['perf/mse'], ylim=(0, 0.1))
        _ = plot_func(data_root=data_root, task_table_name=task, regs=regs, split_keys=['learning_rate'],
                      metrics=['perf/mse'], ylim=(0, 0.1), xlabel='epochs',  ylabel='reward ratio', )
        _ = plot_func(data_root=data_root, task_table_name=task, regs=regs, split_keys=['learning_rate'],
                      metrics=['perf/mse'], ylim=(0, 0.1), xlabel='epochs',  ylabel='reward ratio',
                      shaded_range=False, pretty=True)
