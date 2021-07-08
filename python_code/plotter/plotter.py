from python_code.plotter.plotter_types import get_polar_64_32, get_weighted_polar_64_32, get_polar_128_64, \
    get_weighted_polar_128_64, get_polar_1024_512, get_weighted_polar_1024_512, get_polar_256_128, \
    get_weighted_polar_256_128
from python_code.trainers.fg_trainer import PolarFGTrainer
from python_code.utils.python_utils import load_pkl, save_pkl
from python_code.trainers.trainer import Trainer
from python_code.plotter.plotter_config import *
from dir_definitions import PLOTS_DIR, FIGURES_DIR
import matplotlib.pyplot as plt
from globals import CONFIG
import numpy as np
import datetime
import os


class Plotter:
    def __init__(self, run_over, type):
        self.run_over = run_over
        self.type = type

    def get_fer_plot(self, dec: Trainer, method_name: str):
        print(method_name)
        # set the path to saved plot results for a single method (so we do not need to run anew each time)
        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)
        file_name = '_'.join([method_name])
        plots_path = os.path.join(PLOTS_DIR, file_name + '.pkl')
        print(plots_path)
        # if plot already exists, and the run_over flag is false - load the saved plot
        if os.path.isfile(plots_path) and not self.run_over:
            print("Loading plots")
            saved_dict = load_pkl(plots_path)
            graph = saved_dict[self.type]
        else:
            # otherwise - run again
            print("calculating fresh")
            ber_total, fer_total = dec.evaluate()
            to_save_dict = {'BER': ber_total, 'FER': fer_total}
            save_pkl(plots_path, to_save_dict)
            graph = to_save_dict[self.type]
        return graph

    def plot(self, graph_params, config_params):
        val_SNRs = np.linspace(CONFIG.val_SNR_start, CONFIG.val_SNR_end, num=CONFIG.val_num_SNR)
        # set all parameters based on dict
        for k, v in config_params.items():
            CONFIG.set_value(k, v)
        dec = PolarFGTrainer()
        fer = self.get_fer_plot(dec, graph_params['label'])
        plt.plot(val_SNRs, fer,
                 label=graph_params['label'],
                 color=graph_params['color'],
                 marker=graph_params['marker'])
        plt.title(f'{self.type} Comparison')
        plt.yscale('log')
        plt.xlabel("Eb/N0(dB)")
        plt.ylabel(self.type)
        plt.grid(True, which='both')
        plt.legend(loc='lower left', prop={'size': 15})
        plt.xlim((val_SNRs[0] - 0.5, val_SNRs[-1] + 0.5))

    def plot_curve(self, fer, graph_params):
        val_SNRs = np.linspace(CONFIG.val_SNR_start, CONFIG.val_SNR_end, num=CONFIG.val_num_SNR)
        plt.plot(val_SNRs, fer,
                 label=graph_params['label'],
                 color=graph_params['color'],
                 marker=graph_params['marker'])
        plt.title(f'{self.type} Comparison')
        plt.yscale('log')
        plt.xlabel("Eb/N0(dB)")
        plt.ylabel(self.type)
        plt.grid(True, which='both')
        plt.legend(loc='lower left', prop={'size': 15})
        plt.xlim((val_SNRs[0] - 0.5, val_SNRs[-1] + 0.5))


if __name__ == '__main__':
    plotter = Plotter(run_over=True, type='FER')
    plotter.plot(*get_polar_256_128())
    plotter.plot(*get_weighted_polar_256_128())

    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, 'figure.png'), bbox_inches='tight')
    plt.show()
