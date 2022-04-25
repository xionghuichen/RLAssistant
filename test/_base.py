import unittest
import shutil
import os

class BaseTest(unittest.TestCase):
    """
    Base test class.
    """
    SOURCE_DATA_ROOT = './test_data_root'
    TARGET_DATA_ROOT = './target_data_root'
    TASK_NAME = 'demo_task'
    def remove_and_copy_data(self):
        """
        reset the experiment data for test.
        """
        if os.path.exists(self.TARGET_DATA_ROOT):
            shutil.rmtree(self.TARGET_DATA_ROOT)
        shutil.copytree(self.SOURCE_DATA_ROOT, self.TARGET_DATA_ROOT)
