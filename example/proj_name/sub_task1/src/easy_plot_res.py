from RLA.easy_plot.plot_func import plot_res_func
from RLA.easy_log.const import LOG
import datetime
import os
print(os.getcwd())
prefix_dir = "../log/demo_task"
demo_reg1 = "2021/02/02/19-07-13*"
demo_reg2 = "2021/02/02/19-08-23*"
# filter the experiment name.
# regex_of_your_log_date = str(tester.record_date.strftime("%Y/%m/%d/%H-%M")) + '*env_id=Hopper-v4*'
plot_res_func(prefix_dir, regs=[demo_reg1, demo_reg2],
              param_keys=[], value_keys=["ma/ma_var2"],
              smooth_step=5, replace_legend_keys=["A-1", "A-2"],
              save_name="demo_res.png")
