import os
import os.path as osp

import seaborn as sns
sns.set_style('darkgrid', {'legend.frameon': True})

import matplotlib.pyplot as plt
from RLA.easy_log.tester import exp_manager
from RLA.easy_log.time_step import time_step_holder
from typing import Callable
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
    def pretty_plot_wrapper(cls, name:str, plot_func:Callable,
                            cover=False, legend_outside=False, xlabel='', ylabel='', title='',
                            add_timestamp=True, *args, **kwargs):
        """
        Save the customized plot figure to the RLA database.

        :param name:  file name to save.
        :type name: str
        :param plot_func: the function to plot figures
        :type plot_func: function
        :param cover: if you would like to cover the original figure with the same name, you can set cover to True
        :type cover: bool
        :param legend_outside: let legend be outside of the figure.
        :type legend_outside: bool
        :param xlabel: name of xlabel
        :type xlabel: str
        :param ylabel: name of xlabel
        :type ylabel: str
        :param title: title of the plotted figure
        :type title: str
        :param add_timestamp: add the timestamp (recorded by the timestep holder) to the name.
        :type add_timestamp:  str
        :param args: other parameters to plt.savefig
        :type args:
        :param kwargs:  other parameters to plt.savefig
        :type kwargs:
        :return:
        :rtype:
        """
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