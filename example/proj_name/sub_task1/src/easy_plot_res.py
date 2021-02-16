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

# optional fn to customize your legend
def key_to_legend_fn(key_list, y_name):
    return "+".join(key_list) + "-custom"

plot_res_func(prefix_dir, regs=[demo_reg1, demo_reg2],
              param_keys=["env_id"], value_keys=["ma/ma_var2"],
              smooth_step=5,
              key_to_legend_fn=key_to_legend_fn,
              save_name="demo_res1.png")

plot_res_func(prefix_dir, regs=[demo_reg1, demo_reg2],
              param_keys=[], value_keys=["ma/ma_var2"],
              smooth_step=5, replace_legend_keys=["A-1", "A-2"],
              save_name="demo_res2.png")
