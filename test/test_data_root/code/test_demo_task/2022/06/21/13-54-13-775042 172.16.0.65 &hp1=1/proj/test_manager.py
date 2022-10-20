from test._base import BaseTest
from RLA.easy_log.tester import exp_manager
import os


class ManagerTest(BaseTest):
    def _init_proj(self, hp):
        task_name = 'test_demo_task'
        rla_data_root = '../../test_data_root'
        exp_manager.configure(task_name, private_config_path='../rla_config.yaml', data_root=rla_data_root)
        exp_manager.log_files_gen()
        exp_manager.print_args()

    def test_proj_init(self):
        hp = {
            'hp1': 1,
            'hp2': 2,
        }
        exp_manager.set_hyper_param(**hp)
        exp_manager.add_record_param(['hp1'])
        self._init_proj(hp)