
import argparse
from .config import *
import config
import os.path as osp
from RLA.easy_plot.plot_func import plot_res_func
import matplotlib.pyplot as plt


def argsparser():
    parser = argparse.ArgumentParser("Delete Log")
    parser.add_argument('--log_root', type=str, default=LOG_ROOT)
    # reduce setting
    parser.add_argument('--sub_proj', type=str, default=DEFAULT_PROJ)
    parser.add_argument('--task', type=str, default=DEFAULT_TASK)
    parser.add_argument('--regs', type=str, nargs="+")
    parser.add_argument('--split_keys', type=str, nargs="+", default='')
    parser.add_argument('--scale', type=int, nargs="+", default=[10])
    parser.add_argument('--scale_index', type=int, nargs="+", default=[0])
    parser.add_argument('--misc', type=str, nargs="+", default=DEFAULT_MISC)
    parser.add_argument('--save_root', type=str, default=DEFAULT_SAVE_ROOT)

    args = parser.parse_args()
    return args


# prefix_dir = './seq_gan_imitation/log/2019/'
# regex_strs = ['08/23/23-54-*', '08/24/00-5*']
# split_keys = ['reuse_sample']
# qualities = ['acc/adjusted_r2', "acc/accurancy_generator", "acc/state_distribution_accurancy_generator"]
args = argsparser()
# example: --sub_proj="dfe_sac" --task="HalfCheetah-v2-l2-base-1.5-s-0" --regs="2019/12/06/*" --split_keys="anchor_state_size"
scale_dict = {}
for i in range(len(args.misc)):
    if i in args.scale_index:
        scale_dict[args.misc[i]] = args.scale[args.scale_index.index(i)]
    else:
        scale_dict[args.misc[i]] = 1
prefix_dir = osp.join(args.log_root, args.sub_proj, "log", args.task)
plot_res_func(prefix_dir, args.regs, args.split_keys, args.misc, scale_dict, config)
# regex_strs = "[||08--23--23-54-*||, ||08--24--00-5*||]"
# split_keys = "[||reuse_sample||]"
# qualities = "[||acc--adjusted_r2||]"
# plot_res_func(prefix_dir, regex_strs, split_keys, qualities)
plt.show()