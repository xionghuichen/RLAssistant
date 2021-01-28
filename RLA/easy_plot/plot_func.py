import glob
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from RLA.easy_log import logger
from RLA.easy_plot import plot_util
from RLA.const import DEFAULT_X_NAME


def split_by_task(taskpath, param_keys, y_names):
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
    for split_key in param_keys:
        if split_key in key_value.keys():
            parse_list.append(split_key + '=' + key_value[split_key])
            # filter_key_value[split_key] = key_value[split_key]
        else:
            parse_list.append(split_key + '=NF')
    task_split_key = '.'.join(parse_list)
    param_keys = []
    for y_name in y_names:
        param_keys.append(task_split_key + ' eval:' + y_name)
    return param_keys, y_names

    # if y_names is not None:
    #     param_keys = []
    #     for y_name in y_names:
    #         param_keys.append(task_split_key+' eval:' + y_name)
    #     return param_keys, y_names
    # else:
    #     return task_split_key, y_names
    # return '_'.join(value[-3:])

def auto_gen_key_value_name(dict):
    parse_list = []
    for key, value in dict.iterms():
        parse_list.append(key + '=' + value)


def picture_split(taskpath, single_name=None, param_keys=None, y_names=None):
    if single_name is not None:
        return single_name, None
    else:
        return split_by_task(taskpath, param_keys, y_names)

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


def plot_res_func(prefix_dir, regs, param_keys,
                  value_keys, misc_scale=None, misc_scale_index=None,
                  replace_legend_keys=None,
                  save_name=None,
                  resample=int(1e3), smooth_step=1.0,
                  ylabel=None, x_bound=None, y_bound=None, x_start=None, use_buf=False, remove_outlier=False, xlabel=None,
                  *args, **kwargs):
    dirs = []
    if xlabel is None:
        xlabel = DEFAULT_X_NAME
    if replace_legend_keys is not None:
        assert (len(replace_legend_keys) == len(regs) and len(value_keys) == 1) or \
               (len(regs) == 1 and len(value_keys) == len(replace_legend_keys))
        print("In manual legend-key mode, the number of keys should be one-to-one matched with regs")
    for regex_str in regs:
        print("check regs {}. log found: ".format(osp.join(prefix_dir, regex_str)))
        log_found = glob.glob(osp.join(prefix_dir, regex_str))
        dirs.extend(log_found)
        # print("regex str :{}. log found".format(regex_str))
        for log in log_found:
            print(log)

    results = plot_util.load_results(dirs, names=value_keys + [xlabel], x_bound=[xlabel, x_bound], use_buf=use_buf)

    print("---- load dataset {}---- ".format(len(results)))

    y_names = value_keys # []
    if ylabel is None:
        ylabel = value_keys
    scale_dict = {}
    if misc_scale_index is None:
        misc_scale_index = []
    for i in range(len(value_keys)):
        if i in misc_scale_index:
            scale_dict[value_keys[i]] = misc_scale[misc_scale_index.index(i)]
        else:
            scale_dict[value_keys[i]] = 1
    _, _, lgd, texts = plot_util.plot_results(results, xy_fn= lambda r, y_names: csv_to_xy(r, DEFAULT_X_NAME, y_names,
                                                                                           scale_dict, x_start=x_start, y_bound=y_bound,
                                                                                           remove_outlier=remove_outlier),
                           # xy_fn=lambda r: ts2xy(r['monitor'], 'info/TimestepsSoFar', 'diff/driver_1_2_std'),
                           # split_fn=lambda r: picture_split(taskpath=r, param_keys=param_keys, y_names=y_names)[0],
                           group_fn=lambda r: picture_split(taskpath=r, param_keys=param_keys, y_names=y_names), # picture_split(taskpath=r, y_names=y_names),
                           average_group=True, resample=resample, smooth_step=smooth_step,
                           ylabel=ylabel, xlabel=xlabel, replace_legend_keys=replace_legend_keys,
                            *args, **kwargs)
    print("--- complete process ---")
    if save_name is not None:
        import os

        from RLA.easy_log.const import LOG, OTHER_RESULTS
        dir_name = prefix_dir.replace(LOG, OTHER_RESULTS, 1)
        os.makedirs(dir_name, exist_ok=True)

        if lgd is not None:
            plt.savefig(osp.join(dir_name, save_name), bbox_extra_artists=tuple([lgd] + texts), bbox_inches='tight')
        else:
            plt.savefig(osp.join(dir_name, save_name), bbox_extra_artists=tuple(texts), bbox_inches='tight')
        print("saved location: {}".format(osp.join(dir_name, save_name)))
    plt.show()

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

