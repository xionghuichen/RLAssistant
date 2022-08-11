# Created by xionghuichen at 2022/8/10
# Email: chenxh@lamda.nju.edu.cn
from test._base import BaseTest
import numpy as np
from RLA.easy_log.log_tools import DeleteLogTool, Filter
from RLA.easy_log.log_tools import ArchiveLogTool, ViewLogTool
from RLA.easy_log.tester import exp_manager
from RLA import plot_func
import os

class ScriptTest(BaseTest):

    def get_basic_info(self):
        data_root = 'test_data_root'
        task = 'demo_task'
        return data_root, task

    def test_plot_basic(self):
        data_root, task = self.get_basic_info()

        regs = [
            '2022/03/01/21-[12]*'
        ]
        _ = plot_func(data_root=data_root, task_table_name=task, regs=regs, split_keys=['learning_rate'],
                      metrics=['perf/mse'])
        # customize the figure
        _ = plot_func(data_root=data_root, task_table_name=task, regs=regs, split_keys=['learning_rate'],
                      metrics=['perf/mse'], ylim=(0, 0.1))
        _ = plot_func(data_root=data_root, task_table_name=task, regs=regs, split_keys=['learning_rate'],
                      metrics=['perf/mse'], ylim=(0, 0.1), xlabel='epochs',  ylabel='reward ratio', )


    def test_pretty_plot(self):
        data_root, task = self.get_basic_info()

        regs = [
            '2022/03/01/21-[12]*'
        ]
        # save image
        _ = plot_func(data_root=data_root, task_table_name=task, regs=regs, split_keys=['learning_rate'],
                      metrics=['perf/mse'], ylim=(0, 0.1), xlabel='epochs',  ylabel='reward ratio',
                      shaded_range=False, show_number=False, pretty=True)
        _ = plot_func(data_root=data_root, task_table_name=task, regs=regs, split_keys=['learning_rate'],
                      metrics=['perf/mse'], ylim=(0, 0.1), xlabel='epochs',  ylabel='reward ratio',
                      shaded_range=False, pretty=True, save_name='saved_image.png')

    def test_reg_map_mode(self):
        # reg-map mode.
        data_root, task = self.get_basic_info()
        regs = [
            '2022/03/01/21-[12]*learning_rate=0.01*',
            '2022/03/01/21-[12]*learning_rate=0.00*',
        ]
        _ = plot_func(data_root=data_root, task_table_name=task, regs=regs, split_keys=['learning_rate'],
                      metrics=['perf/mse'],  regs2legends=['lr=0.01', 'lr<=0.001'],
                      shaded_range=False, pretty=True)

    def test_customize_legend_name_mode(self):
        data_root, task = self.get_basic_info()
        regs = [
            '2022/03/01/21-[12]*'
        ]

        def my_key_to_legend(parse_dict, split_keys, y_name):

            task_split_key = '.'.join(f'{k}={parse_dict[k]}' for k in split_keys)
            task_split_key = task_split_key.replace('learning_rate', 'α')
            return task_split_key

        _ = plot_func(data_root=data_root, task_table_name=task, regs=regs, split_keys=['learning_rate'],
                      metrics=['perf/mse'],
                      key_to_legend_fn=my_key_to_legend,
                      shaded_range=False, pretty=True, show_number=False)

    def test_post_process(self):
        data_root, task = self.get_basic_info()
        regs = [
            '2022/03/01/21-[12]*'
        ]

        _ = plot_func(data_root=data_root, task_table_name=task, regs=regs, split_keys=['learning_rate'],
                      metrics=['perf/mse'],
                      scale_dict={'perf/mse': lambda x: np.log(x)},
                      ylabel='RMSE',
                      shaded_range=False, pretty=True, show_number=False)
