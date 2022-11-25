import matplotlib.pyplot as plt
import os.path as osp
import json
import os
import numpy as np
import pandas
from collections import defaultdict, namedtuple
import seaborn as sns
from collections import defaultdict, namedtuple
from RLA.easy_log.logger import read_json, read_csv
sns.set_style('darkgrid', {'legend.frameon':True})



def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]

    valid_only: put nan in entries where the full-sized window is not available

    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out

def one_sided_ema(xolds, yolds, low=None, high=None, n=256, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform one-sided (causal) EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    '''

    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
    assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))


    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0 # last unused old index
    sum_y = 0.
    count_y = 0.
    xnews = np.linspace(low, high, n)
    # print("high {}, low {}, n {}, decay {}".format(high, low, n, decay_steps))
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(- 1. / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            if luoi >= len(xolds):
                break
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(- (xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan

    # print("low_counts_threshold {}".format(low_counts_threshold, np.where(ys[count_ys < low_counts_threshold])[0].shape))
    return xnews, ys, count_ys

def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform symmetric EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    '''
    xs, ys1, count_ys1 = one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0)
    _,  ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold=0)
    ys2 = ys2[::-1]
    count_ys2 = count_ys2[::-1]
    count_ys = count_ys1 + count_ys2
    ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    ys[count_ys < low_counts_threshold] = np.nan
    return xs, ys, count_ys

# Result = namedtuple('Result', 'monitor progress dirname metadata hyper_param')
# Result.__new__.__defaults__ = (None,) * len(Result._fields)

class Result:
    def __init__(self, monitor=None, progress=None, dirname=None, metadata=None, hyper_param=None):
        self.monitor = monitor
        self.progress = progress
        self.dirname = dirname
        self.metadata = metadata
        self.hyper_param = hyper_param

def word_replace(string):
    return string.replace('/', '--').replace("\'", "||")


def word_replace_back(strings):
    return eval(strings.replace('--', '/').replace("||", "\'"))

def load_results(root_dir_or_dirs, names, x_bound, enable_progress=True, use_buf=False, verbose=False):
    '''
    load summaries of runs from a list of directories (including subdirectories)
    Arguments:

    enable_progress: bool - if True, will attempt to load data from progress.csv files (data saved by logger). Default: True


    verbose: bool - if True, will print out list of directories from which the data is loaded. Default: False


    Returns:
    List of Result objects with the following fields:
         - dirname - path to the directory data was loaded from
         - metadata - run metadata (such as command-line arguments and anything else in metadata.json file
         - progress - if enable_progress is True, this field contains pandas dataframe with loaded progress.csv file
    '''
    import re
    if isinstance(root_dir_or_dirs, str):
        rootdirs = [osp.expanduser(root_dir_or_dirs)]
    else:
        rootdirs = [osp.expanduser(d) for d in root_dir_or_dirs]
    allresults = []
    for rootdir in rootdirs:
        assert osp.exists(rootdir), "%s doesn't exist"%rootdir
        for dirname, dirs, files in os.walk(rootdir):
            if '-proc' in dirname:
                files[:] = []
                continue
            if set(['metadata.json', 'progress.json', 'progress.csv']).intersection(files):  # also match monitor files like 0.1.monitor.csv
                # used to be uncommented, which means do not go deeper than current directory if any of the data files
                # are found
                # dirs[:] = []
                result = {'dirname' : dirname}
                if "metadata.json" in files:
                    with open(osp.join(dirname, "metadata.json"), "r") as fh:
                        result['metadata'] = json.load(fh)
                progjson = osp.join(dirname, "progress.json")
                progcsv = osp.join(dirname, "progress.csv")
                if enable_progress:
                    if osp.exists(progjson):
                        result['progress'] = pandas.DataFrame(read_json(progjson))
                    elif osp.exists(progcsv):
                        try:
                            import pandas as pd
                            import csv
                            encode_names = str([name[:4] + name[-4:] for name in names])
                            encode_names = word_replace(encode_names)
                            buf_csv = osp.join(dirname, "progress-{}.csv".format(encode_names))
                            if osp.exists(buf_csv) and use_buf:
                                print("read buf: {}".format(buf_csv))
                                raw_df = read_csv(buf_csv)
                            else:
                                reader = pd.read_csv(progcsv, chunksize=100000,  quoting=csv.QUOTE_NONE,
                                                     encoding='utf-8', index_col=False, comment='#')
                                raw_df = pd.DataFrame()

                                for chunk in reader:
                                    slim_chunk = chunk
                                    # if set(names).issubset(slim_chunk.columns):
                                    existed_names = []
                                    for name in names:
                                        if name not in slim_chunk.columns:
                                            print("[error keys]: {}".format(name))
                                        else:
                                            existed_names.append(name)
                                    if len(existed_names) == 0:
                                        raise RuntimeError("all value_keys cannot be found.")
                                    slim_chunk = slim_chunk[existed_names]
                                    if x_bound[1] is not None:
                                        if isinstance(x_bound[1], tuple):
                                            slim_chunk = slim_chunk[np.logical_and(slim_chunk[x_bound[0]] < x_bound[1][1],
                                                                                   slim_chunk[x_bound[0]] > x_bound[1][0])]
                                        else:
                                            slim_chunk = slim_chunk[slim_chunk[x_bound[0]] < x_bound[1]]
                                    raw_df = pd.concat([raw_df, slim_chunk], ignore_index=True)
                                import csv
                                raw_df.to_csv(buf_csv, index=False)
                            result['progress'] = raw_df
                        except pandas.errors.EmptyDataError:
                            print('skipping progress file in ', dirname, 'empty data')
                        except Exception as e:
                            print("other read error :", e)

                    else:
                        if verbose: print('skipping %s: no progress file'%dirname)


                if result.get('progress') is not None:
                    allresults.append(Result(**result))
                    if verbose:
                        print('successfully loaded %s'%dirname)

    if verbose: print('loaded %i results'%len(allresults))
    return allresults

COLORS = ['blue', 'green', 'red',  'm', 'darkorange', 'k',
          'dodgerblue', 'darkturquoise', 'deeppink', 'brown', 'rosybrown', 'sandybrown',  'gold',
          'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown',    'lightblue', 'lime', 'lavender', 'turquoise',
         'tan', 'salmon',   'darkred', 'darkblue',  'gold']
PRETTY_COLORS = ['orangered',  'royalblue', 'forestgreen',  'orange', 'deeppink', 'deepskyblue']

def default_xy_fn(r, y_name):
    x = np.cumsum(r.monitor.l)
    y = smooth(r.monitor.r, radius=10)
    return x, y

def default_split_fn(r):
    import re
    # match name between slash and -<digits> at the end of the string
    # (slash in the beginning or -<digits> in the end or either may be missing)
    match = re.search(r'[^/-]+(?=(-\d+)?\Z)', r.dirname)
    if match:
        return match.group(0)

def plot_results(
    allresults, *,
    xy_fn=default_xy_fn,
    metrics=None,
    split_fn=None, # default_split_fn,
    group_fn=None, # default_split_fn,
    average_group=False,
    shaded_std=False,
    shaded_err=True,
    shaded_range=True,
    figsize=None,
    legend_outside=True,
    resample=0,
    vary_len_plot=False,
    smooth_step=1.0,
    tiling='symmetric',
    xlabel=None,
    ylabel=None,
    title=None,
    regs2legends=None,
    pretty=False,
    bound_line=None,
    colors=None,
    log=False,
    ylim=None,
    xlim=None,
    show_number=True,
    skip_legend=False,
    split_by_metrics=False,
    base_dpi=90,
    rescale_idx=None):
    '''
    Plot multiple Results objects

    xy_fn: function Result -> x,y           - function that converts results objects into tuple of x and y values.
                                              By default, x is cumsum of episode lengths, and y is episode rewards

    split_fn: function Result -> hashable   - function that converts results objects into keys to split curves into sub-panels by.
                                              That is, the results r for which split_fn(r) is different will be put on different sub-panels.
                                              By default, the portion of r.dirname between last / and -<digits> is returned. The sub-panels are
                                              stacked vertically in the figure.

    group_fn: function Result -> hashable   - function that converts results objects into keys to group curves by.
                                              That is, the results r for which group_fn(r) is the same will be put into the same group.
                                              Curves in the same group have the same color (if average_group is False), or averaged over
                                              (if average_group is True). The default value is the same as default value for split_fn

    average_group: bool                     - if True, will average the curves in the same group and plot the mean. Enables resampling
                                              (if resample = 0, will use 512 steps)

    shaded_std: bool                        - if True (default), the shaded region corresponding to standard deviation of the group of curves will be
                                              shown (only applicable if average_group = True)

    shaded_err: bool                        - if True (default), the shaded region corresponding to error in mean estimate of the group of curves
                                              (that is, standard deviation divided by square root of number of curves) will be
                                              shown (only applicable if average_group = True)

    figsize: tuple or None                  - size of the resulting figure (including sub-panels). By default, width is 6 and height is 6 times number of
                                              sub-panels.


    legend_outside: bool                    - if True, will place the legend outside of the sub-panels.

    resample: int                           - if not zero, size of the uniform grid in x direction to resample onto. Resampling is performed via symmetric
                                              EMA smoothing (see the docstring for symmetric_ema).
                                              Default is zero (no resampling). Note that if average_group is True, resampling is necessary; in that case, default
                                              value is 512.

    smooth_step: float                      - when resampling (i.e. when resample > 0 or average_group is True), use this EMA decay parameter (in units of the new grid step).
                                              See docstrings for decay_steps in symmetric_ema or one_sided_ema functions.


    '''
    score_results = {}
    if vary_len_plot:
        assert resample <= 0, "plot varied length averaged lines only allowed in unresample mode."
    if colors is None:
        if pretty:
            colors = PRETTY_COLORS
        else:
            colors = COLORS

    if pretty:
        assert not split_by_metrics or len(metrics) == 1, "pretty mode cannot support the multiply metric plotting. Please use only one metric for plotting."
    split_by_metrics = split_by_metrics and len(metrics) != 1
    if split_fn is None: split_fn = lambda _ : ''
    if group_fn is None: group_fn = lambda _ : ''
    sk2r = defaultdict(list) # splitkey2results
    if split_by_metrics:
        for y in metrics:
            sk2r[y].extend(allresults)
    else:
        for result in allresults:
            splitkey = split_fn(result)
            sk2r[splitkey].append(result)
    assert len(sk2r) > 0
    assert isinstance(resample, int), "0: don't resample. <integer>: that many samples"
    if tiling == 'vertical' or tiling is None:
        nrows = len(sk2r)
        ncols = 1
    elif tiling == 'horizontal':
        ncols = len(sk2r)
        nrows = 1
    elif tiling == 'symmetric':
        import math
        N = len(sk2r)
        largest_divisor = 1
        for i in range(1, int(math.sqrt(N))+1):
            if N % i == 0:
                largest_divisor = i
        ncols = largest_divisor
        nrows = N // ncols
    figsize = figsize or (7 * ncols, 6 * nrows)
    # if legend_outside:
    #     figsize = list(figsize)
    #     figsize[0] += 4
    #     figsize = tuple(figsize)
    f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize)
    groups = []
    for results in allresults:
        groups.extend(group_fn(results)[0])
    del allresults
    groups = list(set(groups))
    groups.sort()

    # default_samples = 512
    # if average_group:
    #     resample = resample or default_samples
    lgd = None
    for (isplit, sk) in enumerate(sorted(sk2r.keys())):
        g2l = {}
        g2lf = {}
        g2c = defaultdict(int)
        sresults = sk2r[sk]
        gresults = defaultdict(list)
        idx_row = isplit // ncols
        idx_col = isplit % ncols
        ax = axarr[idx_row][idx_col]
        for result in sresults:
            result_groups, y_names = group_fn(result)
            if split_by_metrics:
                y_names = [sk]
            for group, y_name in zip(result_groups, y_names):
                g2c[group] += 1
                res = xy_fn(result, y_name)
                if res is None:
                    continue
                else:
                    x, y = res
                if x is None: x = np.arange(len(y))
                x, y = map(np.asarray, (x, y))

                if rescale_idx is not None:
                    x = (x * (rescale_idx / x[-1])).astype(np.int)
                    x, x_idx = np.unique(x, return_index=True)
                    y = y[x_idx]
                if average_group:
                    gresults[group].append((x,y))
                else:
                    if resample:
                        x, y, counts = symmetric_ema(x, y, x[0], x[-1], resample, decay_steps=smooth_step)
                    l, = ax.plot(x, y, color=colors[groups.index(group) % len(colors)])
                    g2l[group] = l

        if average_group:
            for group in sorted(groups):
                xys = gresults[group]
                if not any(xys):
                    continue
                color = colors[groups.index(group) % len(colors)]
                origxs = [xy[0] for xy in xys]
                maxlen = max(map(len, origxs))
                minxlen = min(map(len, origxs))
                def allequal(qs):
                    return all((q==qs[0]).all() for q in qs[1:])
                if resample:
                    low = max(x[0] for x in origxs)
                    high = min(x[-1] for x in origxs)
                    usex = np.linspace(low, high, resample)
                    ys = []
                    for (x, y) in xys:
                        ys.append(symmetric_ema(x, y, low, high, resample, decay_steps=smooth_step)[1])
                else:
                    assert allequal([x[:minxlen] for x in origxs]), \
                        'If you want to average unevenly sampled data, set resample=<number of samples you want>'
                    if vary_len_plot:
                        usex = origxs[0]
                        for ox in origxs:
                            if len(ox) > len(usex):
                                usex = ox
                        ys = []
                        for xy in xys:
                            if len(xy[1]) < maxlen:
                                y = np.append(xy[1], np.ones(maxlen - len(xy[1])) * np.nan)
                                ys.append(y)
                            else:
                                ys.append(xy[1])
                    else:
                        usex = origxs[0][:minxlen]
                        ys = [xy[1][:minxlen] for xy in xys]
                ymean = np.nanmean(ys, axis=0)
                ystd = np.nanstd(ys, axis=0)
                ymin = np.nanmin(ys, axis=0)
                ymax = np.nanmax(ys, axis=0)
                ystderr = ystd / np.sqrt(len(ys))
                l, = axarr[idx_row][idx_col].plot(usex, ymean, color=color)
                g2l[group] = l
                if shaded_err:
                    g2lf[group + '-se'] = [ax.fill_between(usex, ymean - ystderr, ymean + ystderr, color=color, alpha=.2), ymean, ystderr]
                if shaded_std:
                    g2lf[group + '-ss'] = [ax.fill_between(usex, ymean - ystd,    ymean + ystd,    color=color, alpha=.2), ymean, ystd]
                if shaded_range:
                    g2lf[group + '-sr'] = [ax.fill_between(usex, ymin,    ymax,    color=color, alpha=.1), ymin, ymax]

        ax.set_title(sk)
        if split_by_metrics:
            ax.set_ylabel(sk)
        if log:
            ax.set_yscale('log')

    # https://matplotlib.org/users/legend_guide.html
    # if not pretty:
    #     plt.tight_layout()
    if any(g2l.keys()):
        if show_number:
            legend_keys = np.array(['%s (%i)' % (g, g2c[g]) for g in g2l] if average_group else g2l.keys())
        else:
            legend_keys = np.array(['%s' % (g) for g in g2l] if average_group else g2l.keys())

        legend_lines = np.array(list(g2l.values()))
        sorted_index = np.argsort(legend_keys)
        legend_keys = legend_keys[sorted_index]
        legend_lines = legend_lines[sorted_index]
        if regs2legends is not None:
            legend_keys = np.array(regs2legends)
            # if replace_legend_sort is not None:
            #     sorted_index = replace_legend_sort
            # else:
            #     sorted_index = np.argsort(legend_keys)
            # assert legend_keys.shape[0] == legend_lines.shape[0], \
            #     "The number of lines is not consistent with the keys"
            # legend_keys = legend_keys[sorted_index]
            # legend_lines = legend_lines[sorted_index]
        if pretty:
            for index, l in enumerate(legend_lines):
                l.update(props={"color": colors[index % len(colors)]})
                original_legend_keys = np.array(['%s' % (g) for g in g2l] if average_group else g2l.keys())
                original_legend_keys = original_legend_keys[sorted_index]
                if shaded_err:
                    res = g2lf[original_legend_keys[index] + '-se']
                    res[0].update(props={"color": colors[index % len(colors)]})
                    print("{}-err : ({:.3f} $\pm$ {:.3f})".format(legend_keys[index], res[1][-1], res[2][-1]))
                    score_results[legend_keys[index]+'-err'] = [res[1][-1], res[2][-1]]
                if shaded_std:
                    res = g2lf[original_legend_keys[index] + '-ss']
                    res[0].update(props={"color": colors[index % len(colors)]})
                    print("{}-std :({:.3f} $\pm$ {:.3f})".format(legend_keys[index], res[1][-1], res[2][-1]))
                    score_results[legend_keys[index]+'-std'] = [res[1][-1], res[2][-1]]
                if shaded_range:
                    res = g2lf[original_legend_keys[index] + '-sr']
                    res[0].update(props={"color": colors[index % len(colors)]})
                    print("{}-range : ({:.3f}, {:.3f})".format(legend_keys[index], res[1][-1], res[2][-1]))
                    score_results[legend_keys[index]+'-range'] = [res[1][-1], res[2][-1]]

        if bound_line is not None:
            for bl in bound_line:
                y = np.ones(x.shape) * bl[0]
                l, = ax.plot(x, y, bl[2], color=bl[1])
                legend_lines = np.append(legend_lines, l)
                legend_keys = np.append(legend_keys, bl[3])

        if not skip_legend:
            # print(nrows)
            if len(sk2r.keys()) == 1:
                lgd = ax.legend(
                    legend_lines,
                    legend_keys,
                    loc=2 if legend_outside else None,
                    bbox_to_anchor=(1,1) if legend_outside else None,
                    fontsize=15 if pretty else None)
            else:
                lgd = f.legend(
                    legend_lines,
                    legend_keys,
                    loc='lower center' if legend_outside else None,
                    bbox_to_anchor=(0.5, 0.0) if legend_outside else None,
                    fontsize=15 if pretty else None,
                    borderaxespad=0)
        else:
            lgd = None


    # add xlabels, but only to the bottom row
    if xlabel is not None:
        for ax in axarr[-1]:
            plt.sca(ax)
            if pretty:
                plt.xlabel(xlabel, fontsize=20)
            else:
                plt.xlabel(xlabel)
    # add ylabels, but only to left column
    if ylabel is not None:
        # for ax in axarr[:,0]:
        # plt.sca(axarr[0, 0])
        # plt.ylabel(ylabel)
        # plt.sca(f)
        if pretty:
            f.supylabel(ylabel, fontsize=20, horizontalalignment='center')
        else:
            if not split_by_metrics or len(metrics) == 1:
                f.supylabel(ylabel, horizontalalignment='center')
    if title is not None:
        plt.title(title)
    plt.grid(True)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    texts = []
    if pretty:
        from matplotlib.ticker import ScalarFormatter
        xfmt = ScalarFormatter(useMathText=True)
        xfmt.set_powerlimits((-4, 4))  # Or whatever your limits are . . .
        plt.gca().yaxis.set_major_formatter(xfmt)
        plt.gca().yaxis.offsetText.set_fontsize(15)
        plt.gca().xaxis.set_major_formatter(xfmt)
        plt.gca().xaxis.offsetText.set_fontsize(15)
        # plt.xlabel(xlabel, fontsize=20)
        # plt.ylabel(ylabel, fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(title, fontsize=18)
    else:
        plt.gcf().subplots_adjust(bottom=0.12, left=0.12)
    return f, axarr, lgd, texts, g2lf, score_results

def regression_analysis(df):
    xcols = list(df.columns.copy())
    xcols.remove('score')
    ycols = ['score']
    import statsmodels.api as sm
    mod = sm.OLS(df[ycols], sm.add_constant(df[xcols]), hasconst=False)
    res = mod.fit()
    print(res.summary())

def test_smooth():
    norig = 100
    nup = 300
    ndown = 30
    xs = np.cumsum(np.random.rand(norig) * 10 / norig)
    yclean = np.sin(xs)
    ys = yclean + .1 * np.random.randn(yclean.size)
    xup, yup, _ = symmetric_ema(xs, ys, xs.min(), xs.max(), nup, decay_steps=nup/ndown)
    xdown, ydown, _ = symmetric_ema(xs, ys, xs.min(), xs.max(), ndown, decay_steps=ndown/ndown)
    xsame, ysame, _ = symmetric_ema(xs, ys, xs.min(), xs.max(), norig, decay_steps=norig/ndown)
    plt.plot(xs, ys, label='orig', marker='x')
    plt.plot(xup, yup, label='up', marker='x')
    plt.plot(xdown, ydown, label='down', marker='x')
    plt.plot(xsame, ysame, label='same', marker='x')
    plt.plot(xs, yclean, label='clean', marker='x')
    plt.legend()
    plt.show()