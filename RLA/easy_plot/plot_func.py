import glob
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from RLA.easy_log import logger
from RLA.easy_plot import plot_util
from RLA.const import DEFAULT_X_NAME

def split_by_task(taskpath, split_keys, y_names):
    pair_delimiter = '&'
    kv_delimiter = '='
    pairs = taskpath.dirname.split(pair_delimiter)
    # value = []
    key_value = {}
    for p in pairs:
        key = kv_delimiter.join(p.split(kv_delimiter)[:-1])
        key_value[key] = p.split(kv_delimiter)[-1]
    # filter_key_value = {}
    parse_list = []
    for split_key in split_keys:
        if split_key in key_value.keys():
            parse_list.append(split_key + '=' + key_value[split_key])
            # filter_key_value[split_key] = key_value[split_key]
        else:
            parse_list.append(split_key + '=NF')
    task_split_key = '.'.join(parse_list)
    split_keys = []
    for y_name in y_names:
        split_keys.append(task_split_key + ' eval:' + y_name)
    return split_keys, y_names

    # if y_names is not None:
    #     split_keys = []
    #     for y_name in y_names:
    #         split_keys.append(task_split_key+' eval:' + y_name)
    #     return split_keys, y_names
    # else:
    #     return task_split_key, y_names
    # return '_'.join(value[-3:])

def auto_gen_key_value_name(dict):
    parse_list = []
    for key, value in dict.iterms():
        parse_list.append(key + '=' + value)


def picture_split(taskpath, single_name=None, split_keys=None, y_names=None):
    if single_name is not None:
        return single_name, None
    else:
        return split_by_task(taskpath, split_keys, y_names)

def csv_to_xy(r, x_name, y_name, scale_dict, x_bound=None, x_start=None, y_bound=None, remove_outlier=False):

    df = r.progress.copy().reset_index() # ['progress']
    if df is None:
        logger.warn("empty df!")
        return [], []
    if y_name not in list(df.columns):
        return None
    df.drop(df[np.isnan(df[x_name])].index, inplace=True)
    df.drop(df[np.isnan(df[y_name])].index, inplace=True)
    # pd = pd.dropna(axis=0, how='any')
    x = df[x_name]
    y = df[y_name]
    if x_bound is None:
        x_bound = x.max()
    if x_start is None:
        x_start = x.min()
    filter_index = (x <= x_bound) & (x >= x_start)
    x = x[filter_index]
    y = y[filter_index]
    if y_bound is not None:
        y[y > y_bound] = y_bound
    if remove_outlier:
        z_score = (y - y.mean()) / y.std()
        filter_index = z_score < 10.0
        x = x[filter_index]
        y = y[filter_index]

    y = y * scale_dict[y_name]
    return x, y

def word_replace(string):
    return string.replace('/', '--').replace("\'", "||")

def word_replace_back(strings):
    return eval(strings.replace('--', '/').replace("||", "\'"))

def plot_res_func(prefix_dir, regex_strs, split_keys, qualities, scale_dict, private_config_path, save_name=None, replace=False, resample=int(1e3),
                  smooth_step=1.0, replace_legend_keys=None, ylabel=None, x_bound=None, y_bound=None, x_start=None, legend_outside=True,
                    use_buf=False,
                    remove_outlier=False,
                    xlabel=None, *args, **kwargs):
    import yaml
    import os

    fs = open(os.path.join(private_config_path, "config.yaml"), encoding="UTF-8")
    private_config = yaml.load(fs, Loader=yaml.FullLoader)
    dirs = []
    if xlabel is None:
        xlabel = DEFAULT_X_NAME

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

    results = plot_util.load_results(dirs, names=qualities + [xlabel],
                                     enable_monitor=False, x_bound=[xlabel, x_bound], use_buf=use_buf)
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


def scale_index_to_dict(measure, scale_index, scale):
    scale_dict = {}
    for i in range(len(measure)):
        if i in scale_index:
            scale_dict[measure[i]] = scale[scale_index.index(i)]
        else:
            scale_dict[measure[i]] = 1
    return scale_dict


def show_plt():
    plt.show()

