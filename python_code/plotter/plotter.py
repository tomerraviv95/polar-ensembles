from python_code.plotter.plotter_types import *
from python_code.trainers.fg_trainer import PolarFGTrainer
from python_code.trainers.ensemble_trainer import EnsembleTrainer
from python_code.utils.python_utils import load_pkl, save_pkl
from python_code.trainers.trainer import Trainer
from python_code.plotter.plotter_config import *
from dir_definitions import PLOTS_DIR, FIGURES_DIR, WEIGHTS_DIR
import matplotlib.pyplot as plt
from globals import CONFIG
import numpy as np
import datetime
import os

config_plot_params = {'val_SNR_start' : 1,
                      'val_SNR_end' : 4,
                      'val_num_SNR' : 7,
                      'test_errors' : 500
                }


def get_flops_num(graph_params, config_params, num_of_decoders):
    for k, v in config_params.items():
        CONFIG.set_value(k, v)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    file_name = ''.join(graph_params['label'])
    plots_path = os.path.join(PLOTS_DIR, file_name + '.pkl')
    print(plots_path)
    saved_dict = load_pkl(plots_path)
    BP_fer = saved_dict['FER']
    total_params = 0

    dec = PolarFGTrainer()
    for name, parameter in dec.model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params

    return total_params * (BP_fer * num_of_decoders + 1)


class Plotter:
    def __init__(self, run_over, type):
        self.run_over = run_over
        self.type = type

    def get_fer_plot(self, dec: Trainer, method_name: str, take_crc_0=False):
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
            if self.type == 'CRCPASS':
                return self.plot_crc_passrate(dec)
            if isinstance(dec, EnsembleTrainer):
                ber_total, fer_total = dec.evaluate_test(take_crc_0=take_crc_0)
            else:
                ber_total, fer_total = dec.evaluate()
            to_save_dict = {'BER': ber_total, 'FER': fer_total}
            save_pkl(plots_path, to_save_dict)
            graph = to_save_dict[self.type]
        return graph


    def plot(self, graph_params, config_params, dec_type='FG', take_crc_0=False):
        if config_params["load_weights"]:
            plot_config_path = os.path.join(WEIGHTS_DIR,config_params["run_name"]+"\\config.yaml")
            CONFIG.load_config(plot_config_path)
        # set all parameters based on dict
        for k, v in config_params.items():
            CONFIG.set_value(k, v)
        for k, v in config_plot_params.items():
            CONFIG.set_value(k, v)
        val_SNRs = np.linspace(CONFIG.val_SNR_start, CONFIG.val_SNR_end, num=CONFIG.val_num_SNR)

        if dec_type == 'Ensemble':
            dec = EnsembleTrainer()
        else:
            dec = PolarFGTrainer()

        if self.type == 'CRCPASS':
            self.plot_crc_passrate(dec)
            return

        if self.type == 'JAACARD':
            self.plot_crc_jaacard(dec)
            return

        fer = self.get_fer_plot(dec, graph_params['label'], take_crc_0=take_crc_0)
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


    def get_crc_plot(self, dec: Trainer, method_name: str, type=None):
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
            actual_crc_dist, pred_crc_dist = dec.evaluate_crc_distribution(type=type)
            to_save_dict = {'actual_crc': actual_crc_dist, 'pred_crc': pred_crc_dist}
            save_pkl(plots_path, to_save_dict)
            graph = to_save_dict[self.type]
        return graph

    def plot_crc(self, graph_params, config_params, type=None):
        if config_params["load_weights"]:
            plot_config_path = os.path.join(WEIGHTS_DIR,config_params["run_name"]+"\\config.yaml")
            CONFIG.load_config(plot_config_path)
        # set all parameters based on dict
        for k, v in config_params.items():
            CONFIG.set_value(k, v)
        for k, v in config_plot_params.items():
            CONFIG.set_value(k, v)
        val_SNRs = np.linspace(CONFIG.val_SNR_start, CONFIG.val_SNR_end, num=CONFIG.val_num_SNR)
        dec = PolarFGTrainer()
        crc = self.get_crc_plot(dec, graph_params['label'], type=type)
        bins = 5
        if 'bins' in graph_params.keys():
            bins = graph_params['bins']
        max_val = 2**CONFIG.crc_order
        step = int(2**CONFIG.crc_order/bins)
        if type == 'sum':
            step = 1
            max_val = CONFIG.crc_order + 1

        crc_vals = np.linspace(0, stop=max_val, num=(max_val+1))
        for j,snr in enumerate(val_SNRs):
            plt.figure(num=j)
            crc_tmp = crc[j]
            crc_vals = crc_vals.reshape(np.shape(crc_tmp))
            crc_counts = np.zeros(np.shape(crc_vals))
            if step != 1:
                for i in range(1,crc_vals.size):
                    crc_counts[1+step*int(np.floor(i/step))] += crc_tmp[i]
            else:
                crc_counts = crc_tmp
            plt.bar(crc_vals[1:], crc_counts[1:], width=0.5)
            plt.title(f'{self.type} Comparison @ snr: {snr}')
            plt.yscale('log')
            plt.xlabel("crc val")
            plt.ylabel('counts')
            plt.legend(loc='lower left', prop={'size': 15})
            plt.xlim((crc_vals[0] - 10, crc_vals[-1] + 2))

    def plot_crc_passrate(self, dec):
    # TODO save graph then loads them
        colors = ['orange','blue','green','red','black']
        labels = ['dec 1', 'dec 2', 'dec 3', 'dec 4','designated failed, current not']
        align = [-0.2,-0.1,0,0.1,0.2]
        CONFIG.set_value('test_errors', 5e3)
        dec.model.keep_crc_passrate = True
        dec.model.generateCRCpassrateDict()
        fer,ber = dec.evaluate()
        crc_passrate = dec.model.crc_passrate
        bins = np.array(range(1,len(crc_passrate)))
        plt.figure()

        plt.bar(0.3, 0, width=0.1, color=colors[-1], label=labels[-1]) # lazy label
        for dec_id,res in crc_passrate.items():
            if dec_id == 0:
                continue
            x = res[1:,0] # don't care about the BP
            y = res[1:,1]
            # x = np.array([crc_passrate[i][dec_id][0] for i in range(1,len(crc_passrate))]) # transposed plot
            # y = np.array([crc_passrate[i][dec_id][1] for i in range(1,len(crc_passrate))])
            plt.bar(bins+align[dec_id], x+y, width=0.1, color=colors[-1])
            plt.bar(bins+align[dec_id], x, width=0.1, color=colors[dec_id-1], label=labels[dec_id-1])

        plt.title(f"passed CRC @ snr: {config_plot_params['val_SNR_start']} \n {CONFIG.run_name}")
        plt.xlabel("CRC range id")
        plt.ylabel('counts')
        plt.legend(loc='lower left', prop={'size': 15})


    def plot_crc_jaccard(self, dec):
    # TODO save graph then loads them
        colors = ['orange','blue','green','red']
        labels = ['dec 1', 'dec 2', 'dec 3', 'dec 4']
        align = [-0.2,-0.1,0,0.1,0.2]
        CONFIG.set_value('test_errors', 5e3)
        dec.model.keep_crc_passrate = True
        dec.model.generateCRCpassrateDict()
        fer,ber = dec.evaluate()
        crc_passrate = dec.model.crc_passrate
        bins = np.array(range(1,len(crc_passrate)))
        plt.figure()

        plt.bar(0.3, 0, width=0.1, color=colors[-1], label=labels[-1]) # lazy label
        B = np.zeros(dec.model.num_of_decoders)
        for dec_id, res in crc_passrate.items():
            if dec_id == 0:
                continue
            B[dec_id-1] = res[dec_id,0]
        for dec_id,res in crc_passrate.items():
            if dec_id == 0:
                continue
            A_B_intersection = res[1:,0] # don't care about the BP
            A_without_B = res[1:,1]
            jaccard = A_B_intersection/(A_without_B+B)
            plt.bar(bins+align[dec_id],jaccard,  width=0.1, color=colors[dec_id-1], label=labels[dec_id-1])

        plt.title(f"passed CRC @ snr: {config_plot_params['val_SNR_start']}")
        plt.xlabel("CRC range id")
        plt.ylabel('jaccard')
        plt.legend(loc='lower left', prop={'size': 15})

if __name__ == '__main__':


    ''' 64 32 '''
    # plotter = Plotter(run_over=False, type='BER')
    # plotter.plot(*get_polar_64_32(),dec_type='FG')
    # plotter.plot(*get_weighted_polar_64_32_iter5_crc11(),dec_type='FG')
    # plotter.plot(*get_new_ensemble_64_32_iters5_crc11_sum_decs_4(),dec_type='Ensemble')
    # plotter.plot(*get_new_ensemble_64_32_iters5_crc11_sum_decs_4_best_dec(),dec_type='Ensemble', take_crc_0=True)
    # #

    # #
    # # plotter.plot(*get_weighted_polar_64_32_crc11_iter30(),dec_type='FG')
    #
    # # plotter.plot(*get_ensemble_64_32_iter6_crc11_sum(),dec_type='Ensemble')
    # #
    # # plotter = Plotter(run_over=True, type='FER')
    # plotter.plot(*get_ensemble_64_32_iter6_crc11_sum_mod(),dec_type='Ensemble')

    # plotter = Plotter(run_over=True, type='FER')
    # plotter.plot(*get_ensemble_256_128_crc11_iter6_dec2(), dec_type='Ensemble')
    # plotter.plot(*get_ensemble_256_128_crc11_iter6(),dec_type='Ensemble')
    # plotter.plot(*get_ensemble_256_128_crc11_iter6_dec8(), dec_type='Ensemble')
    # plotter.plot(*get_ensemble_256_128_crc11_iter6_best_dec2(), dec_type='Ensemble', take_crc_0=True)
    # plotter.plot(*get_ensemble_256_128_crc11_iter6_best_dec(),dec_type='Ensemble', take_crc_0=True)
    # plotter.plot(*get_ensemble_256_128_crc11_iter6_best_dec8(), dec_type='Ensemble', take_crc_0=True)
    #plotter.plot(*get_weighted_polar_128_64_crc11_iter24(), dec_type='FG')
   
    ''' 256 128 '''
    # plotter = Plotter(run_over=False, type='FER')
    # plotter.plot(*get_polar_256_128())
    # plotter.plot(*get_weighted_polar_256_128_crc11_iter6())
    # plotter.plot(*get_ensemble_256_128_crc11_iter6(),dec_type='Ensemble')
    # plotter = Plotter(run_over=True, type='FER')
    # plotter.plot(*get_ensemble_256_128_crc11_iter6_best_dec(),dec_type='Ensemble', take_crc_0=True)

    ''' 512 256 '''
    # plotter = Plotter(run_over=False, type='BER')
    # plotter.plot(*get_polar_512_256())
    # plotter.plot(*get_wfg_512_256_iters5_crc11())
    # plotter.plot(*get_ensemble_512_256_iters5_crc11_sum_decs_2(),dec_type='Ensemble')
    # plotter.plot(*get_ensemble_512_256_iters5_crc11_sum_decs_4(),dec_type='Ensemble')
    # plotter.plot(*get_ensemble_512_256_iters5_crc11_sum_decs_6(),dec_type='Ensemble')
    # plotter.plot(*get_ensemble_512_256_iters5_crc11_sum_decs_2_best(),dec_type='Ensemble', take_crc_0=True)
    # plotter.plot(*get_ensemble_512_256_iters5_crc11_sum_decs_4_best(),dec_type='Ensemble', take_crc_0=True)
    # plotter.plot(*get_ensemble_512_256_iters5_crc11_sum_decs_6_best(),dec_type='Ensemble', take_crc_0=True)

    # plotter = Plotter(run_over=True, type='FER')

    ''' 1024 512 '''
    # plotter = Plotter(run_over=False, type='FER')
    #
    # plotter = Plotter(run_over=True, type='FER')
    # plotter.plot(*get_polar_1024_512())
    # plotter.plot(*get_wfg_1024_512_iters5_crc11())
    # plotter.plot(*get_ensemble_1024_512_iters5_crc11_sum_decs_4(),dec_type='Ensemble')
    # plotter.plot(*get_ensemble_1024_512_iters5_crc11_sum_decs_4_best(),dec_type='Ensemble', take_crc_0=True)


    ''' CRC '''
    # plotter = Plotter(run_over=True, type='CRCPASS')
    # plotter.plot(*get_ensemble_512_256_iters5_crc11_sum_decs_6(),dec_type='Ensemble')

    ''' CRC dist '''
    # plotter = Plotter(run_over=True, type='pred_crc')
    # plotter.plot_crc(*get_polar_64_32(), type='sum')



# path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, 'figure.png'), bbox_inches='tight')
    plt.show()
