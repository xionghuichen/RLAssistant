from test._base import BaseTest
from RLA.easy_log.log_tools import DeleteLogTool, Filter
from RLA.easy_log.log_tools import ArchiveLogTool

class ScriptTest(BaseTest):

    def test_delete_reg(self) -> None:
        """
        test delete log filtered by regex.
        """
        self.remove_and_copy_data()
        filter = Filter()
        filter.config(type=Filter.ALL, timstep_bound=1)
        dlt = DeleteLogTool(proj_root=self.TARGET_DATA_ROOT, task_table_name=self.TASK_NAME, regex='2022/03/01/21-13*', filter=filter)
        log_found = dlt.delete_related_log(skip_ask=True)
        assert log_found == 10
        log_found = dlt.delete_related_log(skip_ask=True)
        assert log_found == 0

    def test_delete_reg_small_ts(self):
        """
        test delete log filtered by regex and threshold of time-step.
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

    def test_archive(self):
        self.remove_and_copy_data()
        # archive experiments.
        dlt = ArchiveLogTool(proj_root=self.TARGET_DATA_ROOT, task_table_name=self.TASK_NAME, regex='2022/03/01/21-13*',
                             archive_table_name='archived', remove=False)
        dlt.archive_log(skip_ask=True)
        # remove the archived experiments.
        filter = Filter()
        filter.config(type=Filter.ALL, timstep_bound=1)
        dlt = DeleteLogTool(proj_root=self.TARGET_DATA_ROOT, task_table_name='archived', regex='2022/03/01/21-13*', filter=filter)
        log_found = dlt.delete_related_log(skip_ask=True)
        assert log_found == 10
    