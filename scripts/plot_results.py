
import argparse
import glob
import os.path as osp

from common import plot_util
from common.plot_func import *
from common.private_config import *


def word_replace(string):
    return string.replace('/', '--').replace("\'", "||")

def word_replace_back(strings):
    # string_back = []
    # for string in strings:
    #     string_back.append(string.replace('--', '/').replace("||", "\'"))
    return eval(strings.replace('--', '/').replace("||", "\'"))

def plot_res_func(prefix_dir, regex_strs, split_keys, qualities, scale_dict, save_name=None, replace=False, resample=int(1e3),
                  smooth_step=1.0, replace_legend_keys=None, ylabel=None, x_bound=None, y_bound=None, x_start=None, legend_outside=True,
                use_buf=False,
                    remove_outlier=False,
                    xlabel=DEFAULT_X_NAME, *args, **kwargs):
    dirs = []
    if type(regex_strs) is str and replace:
        regex_strs = word_replace_back(regex_strs)
        split_keys = word_replace_back(split_keys)
        qualities = word_replace_back(qualities)
    for regex_str in regex_strs:
        print("check regs {}. log found: ".format(osp.join(prefix_dir, regex_str)))
        log_found = glob.glob(osp.join(prefix_dir, regex_str))
        dirs.extend(log_found)
        # print("regex str :{}. log found".format(regex_str))
        for log in log_found:
            print(log)

    results = plot_util.load_results(dirs, names=qualities + [DEFAULT_X_NAME],
                                     enable_monitor=False, x_bound=[DEFAULT_X_NAME, x_bound], use_buf=use_buf)
    print("---- load dataset --- ")
    # y_names = ['acc/adjusted_r2', 'acc/accurancy_trans']
    # eval_policy/coupon_avg_rate
    # postfixs = ['coupon_avg_rate', 'coupon_predict_rate', 'roi_avg', 'roi_predict', 'sum_avg_gmv', 'sum_fos',
    #             'sum_predict_gmv', 'sum_spend']
    # old
    # for idx, quality in enumerate(qualities):
    #     y_names = [quality]
    #     # y_names = ['eval_policy/' + postfix, 'eval_real/' + postfix, 'eval_zero/' + postfix]
    #     # split_fn: 我们要对这个指标分成多张子图的时候用的
    #     # group_fn: r 是遍历到的progress.csv的名字， split_keys 是我们要区分的实验组；y_names是我们要评估的指标
    #     plot_util.plot_results(results, xy_fn= lambda r, y_names: csv_to_xy(r, DEFAULT_X_NAME, y_names),
    #                            # xy_fn=lambda r: ts2xy(r['monitor'], 'info/TimestepsSoFar', 'diff/driver_1_2_std'),
    #                            # split_fn=lambda r: picture_split(taskpath=r, split_keys=split_keys, y_names=y_names)[0],
    #                            group_fn=lambda r: picture_split(taskpath=r, split_keys=split_keys, y_names=y_names), # picture_split(taskpath=r, y_names=y_names),
    #                            average_group=True, resample=int(1e3),
    #                            ylabel=quality, xlabel=DEFAULT_X_NAME)
    #     file_name = "r:{},k:{},q:{}.png".format(regex_strs, split_keys, quality)
    #     file_name = word_replace(file_name)
    #
    #     import os
    #     dir_name = "../res/pic/" + prefix_dir[2:] + '/'
    #     os.makedirs(dir_name, exist_ok=True)
    #     plt.savefig(dir_name + file_name)
    #     print("res related location: {}".format("../pic/" + prefix_dir[2:] + file_name))

    y_names = qualities # []
    if ylabel is None:
        ylabel = qualities

    # y_names = ['eval_policy/' + postfix, 'eval_real/' + postfix, 'eval_zero/' + postfix]
    # split_fn: 我们要对这个指标分成多张子图的时候用的
    # group_fn: r 是遍历到的progress.csv的名字， split_keys 是我们要区分的实验组；y_names是我们要评估的指标
    _, _, lgd, texts = plot_util.plot_results(results, xy_fn= lambda r, y_names: csv_to_xy(r, DEFAULT_X_NAME, y_names,
                                                                                           scale_dict, x_bound=x_bound, x_start=x_start, y_bound=y_bound,
                                                                                           remove_outlier=remove_outlier),
                           # xy_fn=lambda r: ts2xy(r['monitor'], 'info/TimestepsSoFar', 'diff/driver_1_2_std'),
                           # split_fn=lambda r: picture_split(taskpath=r, split_keys=split_keys, y_names=y_names)[0],
                           group_fn=lambda r: picture_split(taskpath=r, split_keys=split_keys, y_names=y_names), # picture_split(taskpath=r, y_names=y_names),
                           average_group=True, resample=resample,
                           legend_outside=legend_outside, smooth_step=smooth_step,
                           ylabel=ylabel, xlabel=xlabel, replace_legend_keys=replace_legend_keys,
                            *args, **kwargs)
    print("--- complete process ---")
    file_name = "r:{},k:{},q:{}.pdf".format(regex_strs, split_keys, qualities)
    file_name = word_replace(file_name)
    if save_name is not None:
        import os
        dir_name = "../res/pic/" + prefix_dir[2:] + '/'
        os.makedirs(dir_name, exist_ok=True)
        if lgd is not None:
            plt.savefig(dir_name + save_name, bbox_extra_artists=tuple([lgd] + texts), bbox_inches='tight')
        else:
            plt.savefig(dir_name + save_name, bbox_extra_artists=tuple(texts), bbox_inches='tight')
        print("res related location: {}".format("../pic/" + prefix_dir[2:] + save_name))

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
    parser.add_argument('--misc', type=str, nargs="+", default=DEFAULT_MEASURE)
    parser.add_argument('--save_root', type=str, default=DEFAULT_SAVE_ROOT)


    args = parser.parse_args()
    return args


if __name__=='__main__':



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
    plot_res_func(prefix_dir, args.regs, args.split_keys, args.misc, scale_dict,
                  bound_line=[6000, 'Upper Bound'])
    # regex_strs = "[||08--23--23-54-*||, ||08--24--00-5*||]"
    # split_keys = "[||reuse_sample||]"
    # qualities = "[||acc--adjusted_r2||]"
    # plot_res_func(prefix_dir, regex_strs, split_keys, qualities)
    plt.show()
