import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid', {'legend.frameon':True})
from RLA.easy_log.tester import tester
import os


def simple_scatter(name, datas, texts, pretty=False, xlabel='', ylabel='',
                   cover=False, save_path=None, *args, **kwargs):
    import os.path as osp
    if save_path is None:
        if pretty:
            save_path = osp.join(tester.results_dir, name + '.pdf')
        else:
            save_path = osp.join(tester.results_dir, name + '.png')
    if not osp.exists(save_path) or cover:
        from matplotlib import pyplot as plt
        from matplotlib.ticker import ScalarFormatter
        plt.cla()
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化
        index = 0
        for data, text in zip(datas, texts):
            color = colors[index % len(colors)]
            plt.scatter(data[:, 0], data[:, 1], color=color, marker='x', alpha=0.2)
            plt.annotate(s=str(text), xy=data.mean(axis=0), color=color)
            index += 1
        texts = []
        texts.append(plt.xlabel(xlabel, fontsize=15))
        texts.append(plt.ylabel(ylabel, fontsize=15))
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid()
        ax = plt.gca()  # 获取当前图像的坐标轴信息
        # xfmt = ScalarFormatter(useMathText=True)
        # xfmt.set_powerlimits((-2, 2))  # Or whatever your limits are . . .
        # plt.gca().yaxis.set_major_formatter(xfmt)
        # plt.gcf().subplots_adjust(bottom=0.12, left=0.12)
        # plt.title(name, fontsize=7)
        save_dir = '/'.join(save_path.split('/')[:-1])
        os.makedirs(save_dir, exist_ok=True)
        #
        plt.savefig(save_path, bbox_extra_artists=tuple(texts), bbox_inches='tight')


def simple_hist(name, data, labels=None, pretty=False, xlabel='', ylabel='', title=None,
                colors=None, styles=None, cover=False, save_path=None,*args, **kwargs):
    import os.path as osp
    if save_path is None:
        if pretty:
            save_path = osp.join(tester.results_dir, name + '.pdf')
        else:
            save_path = osp.join(tester.results_dir, name + '.png')
    if not osp.exists(save_path) or cover:
        from matplotlib import pyplot as plt
        from matplotlib.ticker import ScalarFormatter
        plt.cla()
        if pretty:
            # ['r', 'b'], ['x--', '+-']
            if labels is not None:
                for d, l in zip(data, labels):
                    plt.hist(d, label=l, *args, **kwargs)
            else:
                plt.hist(data,  *args, **kwargs)
        else:
            plt.hist(data, label=labels, *args, **kwargs)
        # plt.tight_layout()
        if labels is not None:
            plt.legend(prop={'size': 13})

        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid()
        ax = plt.gca()  # 获取当前图像的坐标轴信息
        # xfmt = ScalarFormatter(useMathText=True)
        # xfmt.set_powerlimits((-2, 2))  # Or whatever your limits are . . .
        # plt.gca().yaxis.set_major_formatter(xfmt)
        # plt.gcf().subplots_adjust(bottom=0.12, left=0.12)
        plt.title(title, fontsize=7)
        save_dir = '/'.join(save_path.split('/')[:-1])
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)


def simple_plot(name, data=None, x=None, y=None, labels=None, pretty=False, xlabel='', ylabel='',
                colors=None, styles=None, cover=False, title=None, legend_outside=False,
                save_path=None, *args, **kwargs):
    import os.path as osp
    if save_path is None:
        if pretty:
            save_path = osp.join(tester.results_dir, name + '.pdf')
        else:
            save_path = osp.join(tester.results_dir, name + '.png')

    if not osp.exists(save_path) or cover:
        from matplotlib import pyplot as plt
        plt.cla()
        if labels is None:
            if data is not None:
                for d in data:
                    plt.plot(d, *args, **kwargs)
            elif x is not None:
                for x_i, y_i in zip(x, y):
                    plt.plot(x_i, y_i, *args, **kwargs)
            else:
                raise NotImplementedError
        else:
            if pretty:
                # ['r', 'b'], ['x--', '+-']
                if data is not None:
                    for d, l, c, s in zip(data, labels, colors, styles):
                        plt.plot(d, s, label=l, color=c, *args, **kwargs)
                elif x is not None:
                    for x_i, y_i, l, c, s in zip(x, y, labels, colors, styles):
                        plt.plot(x_i, y_i, s, label=l, color=c, *args, **kwargs)
                else:
                    raise NotImplementedError
            else:
                if data is not None:
                    for d, l in zip(data, labels):
                        plt.plot(d, label=l, *args, **kwargs)
                elif x is not None:
                    for x_i, y_i, l_i in zip(x, y, labels):
                        plt.plot(x_i, y_i, label=l_i, *args, **kwargs)
                else:
                    raise NotImplementedError
        if labels is not None:
            # plt.xlabel('time-step (per day)', fontsize=15)
            # plt.ylabel('normalized FOs', fontsize=15)
            lgd = plt.legend(prop={'size': 15},
                             loc=2 if legend_outside else None,
                             bbox_to_anchor=(1, 1) if legend_outside else None,
                             )
        else:
            lgd = None

        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(title, fontsize=13)
        plt.grid(True)
        save_dir = '/'.join(save_path.split('/')[:-1])
        os.makedirs(save_dir, exist_ok=True)
        if lgd is not None:
            plt.savefig(save_path, bbox_extra_artists=tuple([lgd]), bbox_inches='tight')
        else:
            plt.savefig(save_path)
