import os
import os.path as osp

import seaborn as sns
sns.set_style('darkgrid', {'legend.frameon': True})

import matplotlib.pyplot as plt
from RLA.easy_log.tester import exp_manager
from RLA.easy_log.time_step import time_step_holder
# video recorder


# figure recorder
class MatplotlibRecorder:
    @classmethod
    def save(cls, name=None, fig=None, cover=False, add_timestamp=True, **kwargs):
        save_path = osp.join(exp_manager.results_dir, name)
        save_path_split = save_path.split('/')
        if add_timestamp:
            save_path = '/'.join(save_path_split[:-1]) + '/' + str(time_step_holder.get_time()) + "-" + str(save_path_split[-1])
        if not osp.exists(save_path) or cover:
            save_dir = '/'.join(save_path.split('/')[:-1])
            os.makedirs(save_dir, exist_ok=True)
            if fig is not None:
                fig.savefig(save_path, **kwargs)
            else:
                plt.savefig(save_path, **kwargs)

    @classmethod
    def pretty_plot_wrapper(cls, name, plot_func, cover=False, legend_outside=False, xlabel='', ylabel='', title='',
                            add_timestamp=True, *args, **kwargs):
        plt.cla()
        plot_func()
        lgd = plt.legend(prop={'size': 15}, loc=2 if legend_outside else None,
                         bbox_to_anchor=(1, 1) if legend_outside else None)
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(title, fontsize=13)
        plt.grid(True)
        if lgd is not None:
            cls.save(name, cover=cover, add_timestamp=add_timestamp, bbox_extra_artists=tuple([lgd]),
                     bbox_inches='tight', *args, **kwargs)
        else:
            cls.save(name, cover=cover, add_timestamp=add_timestamp, *args, **kwargs)