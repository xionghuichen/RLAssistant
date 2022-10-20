import glob
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from RLA.easy_log import logger
from RLA.easy_plot import plot_util
from RLA.const import DEFAULT_X_NAME
from RLA.easy_log.const import LOG, ARCHIVE_TESTER


def default_key_to_legend(parse_list, y_name):
    task_split_key = '.'.join(parse_list)
    return task_split_key + ' eval:' + y_name

def split_by_task(taskpath, param_keys, y_names, key_to_legend_fn):
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
    param_keys = []
    for y_name in y_names:
        param_keys.append(key_to_legend_fn(parse_list, y_name))
    return param_keys, y_names

    # if y_names is not None:
    #     param_keys = []
    #     for y_name in y_names:
    #         param_keys.append(task_split_key+' eval:' + y_name)
    #     return param_keys, y_names
    # else:
    #     return task_split_key, y_names
    # return '_'.join(value[-3:])

def split_by_reg(taskpath, reg_group, y_names):
    task_split_key = "None"
    for i , reg_k in enumerate(reg_group.keys()):
        if taskpath.dirname in reg_group[reg_k]:
            assert task_split_key == "None", "one experiment should belong to only one reg_group"
            task_split_key = str(i)
    assert len(y_names) == 1
    return task_split_key, y_names

# def split_by_value_key(taskpath, reg_group, y_names):
#     assert len(reg_group) == 1
#     return y_names, y_names


def auto_gen_key_value_name(dict):
    parse_list = []
    for key, value in dict.iterms():
        parse_list.append(key + '=' + value)


def picture_split(taskpath, single_name=None, param_keys=None, y_names=None,
                  key_to_legend_fn=None):
    if single_name is not None:
        return single_name, None
    else:
        return split_by_task(taskpath, param_keys, y_names, key_to_legend_fn=key_to_legend_fn)

def csv_to_xy(r, x_name, y_name, scale_dict, x_bound=None, x_start=None, y_bound=None, remove_outlier=False):

    df = r.progress.copy().reset_index()
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

    y = scale_dict[y_name](y)
    return x, y

def word_replace(string):
    return string.replace('/', '--').replace("\'", "||")

def word_replace_back(strings):
    return eval(strings.replace('--', '/').replace("||", "\'"))


def plot_res_func(prefix_dir:str, regs, param_keys,
                  value_keys, scale_dict=None,
                  replace_legend_keys=None,
                  save_name=None,
                  resample=int(1e3), smooth_step=1.0,
                  ylabel=None, x_bound=None, y_bound=None, x_start=None, use_buf=False,
                  remove_outlier=False, xlabel=None,
                  key_to_legend_fn=None,
                  verbose=True, *args, **kwargs):
    logger.warn("the function is deprecated. please check the plot_func_v2 as the new implementation")
    dirs = []
    if key_to_legend_fn is None:
        key_to_legend_fn = default_key_to_legend
    if xlabel is None:
        xlabel = DEFAULT_X_NAME
    reg_group = {}

    for regex_str in regs:
        if regex_str[0] == '/':
            regex_str = regex_str[1:]
        if verbose:
            print("check regs {}. log found: ".format(osp.join(prefix_dir, regex_str)))

        log_found = glob.glob(osp.join(prefix_dir, regex_str))
        dirs.extend(log_found)
        reg_group[regex_str] = []

        for log in log_found:
            if verbose:
                print(log)
            reg_group[regex_str].append(log)

    results = plot_util.load_results(dirs, names=value_keys + [DEFAULT_X_NAME], x_bound=[DEFAULT_X_NAME, x_bound], use_buf=use_buf)
    if verbose:
        print("---- load dataset {}---- ".format(len(results)))

    y_names = value_keys # []
    if ylabel is None:
        ylabel = value_keys
    final_scale_dict = {}
    # if misc_scale_index is None:
    #     misc_scale_index = []
    for i in range(len(value_keys)):
        final_scale_dict[value_keys[i]] = lambda x: x
    if scale_dict is not None:
        final_scale_dict.update(scale_dict)
    if replace_legend_keys is not None:
        assert len(replace_legend_keys) == len(regs) and len(value_keys) == 1,  \
            "In manual legend-key mode, the number of keys should be one-to-one matched with regs"
        # if len(replace_legend_keys) == len(regs):
        group_fn = lambda r: split_by_reg(taskpath=r, reg_group=reg_group, y_names=y_names)
        # elif len(value_keys) == len(replace_legend_keys):
        #     group_fn = lambda r: split_by_value_key(taskpath=r, reg_group=reg_group, y_names=y_names)
        # else:
        #     raise NotImplementedError
    else:
        group_fn = lambda r: picture_split(taskpath=r, param_keys=param_keys, y_names=y_names,
                                           key_to_legend_fn=key_to_legend_fn)

    _, _, lgd, texts, g2lf, score_results = plot_util.plot_results(results, xy_fn= lambda r, y_names: csv_to_xy(r, DEFAULT_X_NAME, y_names,
                                                                                           final_scale_dict, x_start=x_start, y_bound=y_bound,
                                                                                           remove_outlier=remove_outlier),
                           # xy_fn=lambda r: ts2xy(r['monitor'], 'info/TimestepsSoFar', 'diff/driver_1_2_std'),
                           # split_fn=lambda r: picture_split(taskpath=r, param_keys=param_keys, y_names=y_names)[0],
                           group_fn=group_fn, # picture_split(taskpath=r, y_names=y_names),
                           average_group=True, resample=resample, smooth_step=smooth_step,
                           ylabel=ylabel, xlabel=xlabel, replace_legend_keys=replace_legend_keys,
                            *args, **kwargs)
    print("--- complete process ---")
    if save_name is not None:
        import os

        from RLA.easy_log.const import LOG, OTHER_RESULTS
        dir_name = prefix_dir.replace(f"/{LOG}/", f"/{osp.join(OTHER_RESULTS, 'easy_plot')}/", 1)
        file_name = osp.join(dir_name, save_name)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        if lgd is not None:
            plt.savefig(file_name, bbox_extra_artists=tuple([lgd] + texts), bbox_inches='tight')
        else:
            plt.savefig(file_name, bbox_extra_artists=tuple(texts), bbox_inches='tight')
        print("saved location: {}".format(file_name))
    plt.show()
    return g2lf, score_results


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

