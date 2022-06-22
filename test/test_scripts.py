from test._base import BaseTest
from RLA.easy_log.log_tools import DeleteLogTool, Filter
from RLA.easy_log.log_tools import ArchiveLogTool, ViewLogTool
from RLA.easy_log.tester import exp_manager

import os

class ScriptTest(BaseTest):

    def test_delete_reg(self) -> None:
        """
        test delete log filtered by regex. See rla_scripts/delete_expt.py
        """
        self.remove_and_copy_data()
        filter = Filter()
        filter.config(type=Filter.ALL, timstep_bound=1)
        dlt = DeleteLogTool(proj_root=self.TARGET_DATA_ROOT, task_table_name=self.TASK_NAME, regex='2022/03/01/21-13*', filter=filter)
        log_found = dlt.delete_related_log(skip_ask=True)
        assert log_found == 10
        log_found = dlt.delete_related_log(skip_ask=True)
        assert log_found == 0

    def test_delete_reg_small_ts(self) -> None:
        """
        test delete log filtered by regex and threshold of time-step.  See rla_scripts/delete_expt.py
        """
        self.remove_and_copy_data()
        filter = Filter()
        # none of the experiment satisfied timestep <=1
        filter.config(type=Filter.SMALL_TIMESTEP, timstep_bound=1)
        dlt = DeleteLogTool(proj_root=self.TARGET_DATA_ROOT, task_table_name=self.TASK_NAME, regex='2022/03/01/21-13*', filter=filter)
        log_found = dlt.delete_small_timestep_log(skip_ask=True)
        assert log_found == 0
        # all the experiment satisfied timestep <=2000
        filter.config(type=Filter.SMALL_TIMESTEP, timstep_bound=2000)
        dlt = DeleteLogTool(proj_root=self.TARGET_DATA_ROOT, task_table_name='demo_task', regex='2022/03/01/21-13*', filter=filter)
        log_found = dlt.delete_small_timestep_log(skip_ask=True)
        assert log_found == 10
        # nothing left
        filter.config(type=Filter.SMALL_TIMESTEP, timstep_bound=2000)
        dlt = DeleteLogTool(proj_root=self.TARGET_DATA_ROOT, task_table_name=self.TASK_NAME, regex='2022/03/01/21-13*', filter=filter)
        log_found = dlt.delete_small_timestep_log(skip_ask=True)
        assert log_found == 0

    def test_archive(self) -> None:
        """
        archive experiment log. See rla_scripts/archive_expt.py
        """
        self.remove_and_copy_data()
        # archive experiments.
        dlt = ArchiveLogTool(proj_root=self.TARGET_DATA_ROOT, task_table_name=self.TASK_NAME, regex='2022/03/01/21-13*')
        dlt.archive_log(skip_ask=True)
        # remove the archived experiments.
        filter = Filter()
        filter.config(type=Filter.ALL, timstep_bound=1)
        dlt = DeleteLogTool(proj_root=self.TARGET_DATA_ROOT + '/arc', regex='2022/03/01/21-13*', filter=filter, task_table_name=self.TASK_NAME)
        log_found = dlt.delete_related_log(skip_ask=True)
        assert log_found == 10

    def test_view(self) -> None:
        """
        view experiment log.See rla_scripts/view_expt.py

        """
        self.remove_and_copy_data()
        dlt = ViewLogTool(proj_root=self.TARGET_DATA_ROOT, task_table_name=self.TASK_NAME, regex='2022/03/01/21-13*')
        dlt.view_log(skip_ask=True)