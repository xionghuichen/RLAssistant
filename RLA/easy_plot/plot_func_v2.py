# Created by xionghuichen at 2022/8/10
# Email: chenxh@lamda.nju.edu.cn
import glob
import os.path as osp
import os
import dill
import copy
import numpy as np
from typing import Dict, List, Tuple, Type, Union, Optional, Callable
import matplotlib.pyplot as plt

from RLA import logger
from RLA.const import DEFAULT_X_NAME
from RLA.query_tool import experiment_data_query, extract_valid_index

from RLA.easy_plot import plot_util
from RLA.easy_log.const import LOG, ARCHIVE_TESTER, OTHER_RESULTS



def default_key_to_legend(parse_dict, split_keys, y_name, use_y_name=True):
    task_split_key = '.'.join(f'{k}={parse_dict[k]}' for k in split_keys)
    if use_y_name:
        return task_split_key + ' eval:' + y_name
    else:
        return task_split_key


def plot_func(data_root:str, task_table_name:str, regs:list, split_keys:list, metrics:list,
              use_buf=False, verbose=True,
              xlim: Optional[tuple] = None,
              xlabel: Optional[str] = DEFAULT_X_NAME, ylabel: Optional[str] = None,
              scale_dict: Optional[dict] = None, regs2legends: Optional[list] = None,
              key_to_legend_fn: Optional[Callable] = default_key_to_legend,
              split_by_metrics=True,
              save_name: Optional[str] = None, *args, **kwargs):
    """
    A high-level matplotlib plotter.
    The function is to load your experiments and plot curves.
    You can group several experiments into a single figure through this function.
    It is completed by loading experiments satisfying [data_root, task_table_name, regs] pattern,
    grouping by "split_keys" or by the "regs" terms (see regs2legends), and plotting the customized "metrics".

    The function support several configure to customize the figure, including xlim, xlabel, ylabel, key_to_legend_fn, etc.
    The function also supports several configure to post-process your log data, including resample, smooth_step, scale_dict, key_to_legend_fn, etc.
    The function also supports several configure to beautify the figure, see the parameters of plot_util.plot_results.

    :param data_root:
    :type data_root:
    :param task_table_name:
    :type task_table_name:
    :param regs:
    :type regs:
    :param split_keys:
    :type split_keys:
    :param metrics:
    :type metrics:
    :param use_buf: use the preloaded csv data instead of loading from scratch
    :type use_buf: bool
    :param xlim: set the x limits of the x axes.
    :type xlim: Optional[tuple]
    :param xlabel: set the label of the y axes.
    :type xlabel: Optional[str]
    :param ylabel: set the label of the y axes.
    :type ylabel: Optional[str]
    :param scale_dict: a function dict, to map the value of the metrics through customize functions.
    e.g.,set metrics = ['return'], scale_dict = {'return': lambda x: np.log(x)}, then we will plot a log-scale return.
    :type scale_dict: Optional[dict]
    :param regs2legends: use regex-to-legend mode to plot the figure. Each iterm in regs will be gouped into a curve.
    In this reg2legend_map mode, you should define the lgend name for each curve. See test/test_plot/test_reg_map_mode for details.
    :type regs2legends: Optional[list] = None
    :param key_to_legend_fn: we give a default function to stringify the k-v pairs. you can customize your own function in key_to_legend_fn.
    See default_key_to_legend for the detault way and test/test_plot/test_customize_legend_name_mode for details.
    :type key_to_legend_fn: Optional[Callable] = default_key_to_legend
    :param split_by_metrics: you can plot figure with multiple metrics together.
    By default, we will split the curves with the metric and merge them into a group figure.
    If you would like to print multiple metrics in single figure, please set the parameter to False.
    :type split_by_metrics: Optional[bool]
    :param args/kwargs: send other parameters to plot_util.plot_results
    :return:
    :rtype:
    """
    results = []
    reg_group = {}
    for reg in regs:
        reg_group[reg] = []
        print("searching", reg)
        tester_dict = experiment_data_query(data_root, task_table_name, reg, ARCHIVE_TESTER)
        log_dict = experiment_data_query(data_root, task_table_name, reg, LOG)
        counter = 0
        for k, v in log_dict.items():
            result = plot_util.load_results(v.dirname, names=metrics + [DEFAULT_X_NAME],
                                   x_bound=[DEFAULT_X_NAME, xlim], use_buf=use_buf)
            if len(result) == 0:
                continue
            assert len(result) == 1
            result = result[0]
            if verbose:
                print("find log", v.dirname)
            counter += 1
            result.hyper_param = tester_dict[k].exp_manager.hyper_param
            results.append(result)
            reg_group[reg].append(result)
        print("find log number", counter)
    final_scale_dict = {}
    for m in metrics:
        final_scale_dict[m] = lambda x: x
    if scale_dict is not None:
        final_scale_dict.update(scale_dict)

    y_names = metrics # []
    if ylabel is None:
        ylabel = metrics

    if regs2legends is not None:
        assert len(regs2legends) == len(regs) and len(metrics) == 1,  \
            "In manual legend-key mode, the number of keys should be one-to-one matched with regs"
        # if len(regs2legends) == len(regs):
        group_fn = lambda r: split_by_reg(taskpath=r, reg_group=reg_group, y_names=y_names)
    else:
        group_fn = lambda r: picture_split(taskpath=r, split_keys=split_keys, y_names=y_names,
                                           key_to_legend_fn=lambda parse_dict, split_keys, y_name:
                                           key_to_legend_fn(parse_dict, split_keys, y_name, not split_by_metrics))
    _, _, lgd, texts, g2lf, score_results = \
        plot_util.plot_results(results, xy_fn= lambda r, y_names: csv_to_xy(r, DEFAULT_X_NAME, y_names, final_scale_dict),
                           group_fn=group_fn, average_group=True, ylabel=ylabel, xlabel=xlabel, metrics=metrics,
                               split_by_metrics=split_by_metrics, regs2legends=regs2legends, *args, **kwargs)
    print("--- complete process ---")
    if save_name is not None:
        import os
        file_name = osp.join(data_root, OTHER_RESULTS, 'easy_plot', save_name)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        if lgd is not None:
            plt.savefig(file_name, bbox_extra_artists=tuple([lgd] + texts), bbox_inches='tight')
        else:
            plt.savefig(file_name, bbox_extra_artists=tuple(texts), bbox_inches='tight')
        print("saved location: {}".format(file_name))
    plt.show()
    return g2lf, score_results


def split_by_reg(taskpath, reg_group, y_names):
    task_split_key = "None"
    for i , reg_k in enumerate(reg_group.keys()):
        for result in reg_group[reg_k]:
            if taskpath.dirname == result.dirname:
                assert task_split_key == "None", "one experiment should belong to only one reg_group"
                task_split_key = str(i)
    assert len(y_names) == 1
    return task_split_key, y_names

def split_by_task(taskpath, split_keys, y_names, key_to_legend_fn):
    pair_delimiter = '&'
    kv_delimiter = '='
    parse_dict = {}
    for split_key in split_keys:
        if split_key in taskpath.hyper_param:
            parse_dict[split_key] = str(taskpath.hyper_param[split_key])
            # parse_list.append(split_key + '=' + str(taskpath.hyper_param[split_key]))
        else:
            parse_dict[split_key] = 'NF'
            # parse_list.append(split_key + '=NF')
    param_keys = []
    for y_name in y_names:
        param_keys.append(key_to_legend_fn(parse_dict, split_keys, y_name))
    return param_keys, y_names


def picture_split(taskpath, single_name=None, split_keys=None, y_names=None,
                  key_to_legend_fn=None):
    if single_name is not None:
        return single_name, None
    else:
        return split_by_task(taskpath, split_keys, y_names, key_to_legend_fn=key_to_legend_fn)


def csv_to_xy(r, x_name, y_name, scale_dict, x_bound=None, x_start=None):

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

    y = scale_dict[y_name](y)
    return x, y
