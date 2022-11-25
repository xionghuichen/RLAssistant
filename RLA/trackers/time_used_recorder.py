# Created by xionghuichen at 2022/7/29
# Email: chenxh@lamda.nju.edu.cn
from RLA.easy_log import logger
import time

class SingleTimeTracker:
    def __init__(self,name:str='untitled') -> None:
        self.name=name
        self.t = 0.0
        self.call_time=0
        self.time_cost = 0.0

    def __enter__(self):
        # trace time
        self.call_time+=1            
        self.t = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time_cost += time.perf_counter() - self.t


class TimeTracker:
    def __init__(self):
        self.t0=time.time()#to calc total time
        self.time_dict=dict()

    def add(self,name='untitled'):
        """
        :param name: specify the SingleTimeTracker in the time_dict, recommend use
        line num in the scripts, like 'xxx.py Line xxx', can be easily got in python scripts using 
        `os.path.basename(__file__)+'  line '+str(sys._getframe().f_lineno)`  
        """
        if name not in self.time_dict.keys():
            self.time_dict.update({name:SingleTimeTracker(name)})
        return self.time_dict[name]

    def __call__(self, name:str):
        return self.time_dict[name]
        
    def clear(self):
        self.time_dict=dict()

    def statistic_entry(self,name:str):
        """
        calc total calls of the 
        """
        assert name in self.time_dict.keys()

        t1=time.time()
        t_passed=t1-self.t0
        return {
            'total calls/'+name:self.time_dict[name].call_time,
            'total time cost/'+name:self.time_dict[name].time_cost,
            'average time cost/'+name:self.time_dict[name].time_cost/(1e-6+self.time_dict[name].call_time),
            'time cost percentage/'+name:self.time_dict[name].time_cost/(1e-6+t_passed)
        }

    def get_info(self):
        info={}
        for k in self.time_dict.keys():
            for entry_k,entry_v in self.statistic_entry(k).items():
                info[entry_k]=entry_v
        return info

    def log(self,exclude_lst=['csv']):
        logger.info('---------time dashboard---------')
        for k in self.time_dict.keys():
            for entry_k,entry_v in self.statistic_entry(k).items():
                logger.record_tabular('time_used/'+entry_k,entry_v,exclude=exclude_lst)
                logger.info(f"[{entry_k}]: {entry_v}")
            logger.info('')
        logger.info('---------dashboard end---------')


rc_start_time = {}

def time_record(name):
    """
    record the consumed time of your code snippet. call this function to start a recorder.
    "name" is identifier to distinguish different recorder and record different snippets at the same time.
    call time_record_end to end a recorder.
    :param name: identifier of your code snippet.
    :type name: str
    :return:
    :rtype:
    """
    assert name not in rc_start_time
    rc_start_time[name] = time.time()


def time_record_end(name):
    """
    record the consumed time of your code snippet. call this function to start a recorder.
    "name" is identifier to distinguish different recorder and record different snippets at the same time.
    call time_record_end to end a recorder.
    :param name: identifier of your code snippet.
    :type name: str
    :return:
    :rtype:
    """
    end_time = time.time()
    start_time = rc_start_time[name]
    logger.record_tabular("time_used/{}".format(name), end_time - start_time)
    logger.info("[test] func {0} time used {1:.2f}".format(name, end_time - start_time))
    del rc_start_time[name]