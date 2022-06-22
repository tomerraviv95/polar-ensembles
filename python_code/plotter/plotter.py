from python_code.plotter.plotter_types import *
from python_code.trainers.fg_trainer import PolarFGTrainer
from python_code.trainers.ensemble_trainer import EnsembleTrainer
from python_code.decoders.fg_decoder import FGDecoder
from python_code.utils.python_utils import load_pkl, save_pkl
from python_code.trainers.trainer import Trainer
from python_code.plotter.plotter_config import *
from dir_definitions import PLOTS_DIR, FIGURES_DIR, WEIGHTS_DIR
import matplotlib.pyplot as plt
from globals import CONFIG, DEVICE
import numpy as np
import datetime
import os

config_plot_params = {'val_SNR_start' : 1,
                      'val_SNR_end' : 4,
                      'val_num_SNR' : 7,
                      'test_errors' : 500
                }


def get_flops_num(graph_params, config_params, num_of_decoders):
    ''' function receive plot for BP and decoders num for ensemble
        return the ensemble flops per SNR '''
    for k, v in config_params.items():
        CONFIG.set_value(k, v)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    file_name = ''.join(graph_params['label'])
    plots_path = os.path.join(PLOTS_DIR, file_name + '.pkl')
    print(f"plots path: {plots_path}")

    if not os.path.isfile(plots_path):
        plotter = Plotter(run_over=True, type='FER')
        plotter.plot(graph_params, config_params,dec_type='FG')
        plt.close()

    saved_dict = load_pkl(plots_path)
    BP_fer = saved_dict['FER']
    total_params = 0

    dec = PolarFGTrainer()
    for name, parameter in dec.model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params

    return total_params * (BP_fer * num_of_decoders + 1)

def plot_avg_flops(ens_flops, ens_dec_num, config_params, title=""):
    ''' compare flops for ensemble and WFG with equal iterations and N iterations
    '''

    ens_iters = config_params['iteration_num']
    N = ens_iters*ens_dec_num
    code_len = config_params['code_len']
    info_len=config_params['info_len']
    if not title:
        title = f"Flops comparison: ({code_len},{info_len})"
    val_SNRs = np.linspace(config_plot_params['val_SNR_start'], config_plot_params['val_SNR_end'], num=config_plot_params['val_num_SNR'])

    dec1 = FGDecoder(code_len=code_len, info_len=info_len, design_snr=CONFIG.design_SNR, clipping_val=CONFIG.clipping_val, iteration_num=ens_iters, device=DEVICE)
    dec2 = FGDecoder(code_len=code_len, info_len=info_len, design_snr=CONFIG.design_SNR, clipping_val=CONFIG.clipping_val, iteration_num=N, device=DEVICE)

    total_params1 = 0
    for name, parameter in dec1.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params1 += params

    total_params2 = 0
    for name, parameter in dec2.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params2 += params

    wfg_lowbound_flops = [total_params1]*len(val_SNRs)
    wfg_highbound_flops = [total_params2]*len(val_SNRs)

    plt.plot(val_SNRs, ens_flops, label=f"Ensemble {ens_iters} iterations {ens_dec_num} decoders", color="green", marker="*")
    plt.plot(val_SNRs, wfg_lowbound_flops, label=f"WBP {ens_iters} iterations", color="red", marker="o")
    plt.plot(val_SNRs, wfg_highbound_flops, label=f"WBP {N} iterations", color="cyan", marker="o")
    plt.title(title)
    plt.yscale('log')
    plt.xlabel("Eb/N0(dB)")
    plt.ylabel("Average Flops")
    plt.grid(True, which='both')
    plt.legend(loc='lower left', prop={'size': 15})






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


    def get_crc_plot(self, dec: Trainer, method_name: str, type=None, only_crc_errors=True):
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
            actual_crc_dist, pred_crc_dist = dec.evaluate_crc_distribution(type=type, only_crc_errors=only_crc_errors)
            to_save_dict = {'actual_crc': actual_crc_dist, 'pred_crc': pred_crc_dist}
            save_pkl(plots_path, to_save_dict)
            graph = to_save_dict[self.type]
        return graph

    def plot_crc(self, graph_params, config_params, type=None, only_crc_errors=True, words_count=1e5):
        '''only crc errors will keep only crc vals that are not 0
            evaluating CRC in trainer.evaluate_crc_dist()
            words count will be the number of CRC evaluated through channel
        '''
        if config_params["load_weights"]:
            plot_config_path = os.path.join(WEIGHTS_DIR,config_params["run_name"]+"\\config.yaml")
            CONFIG.load_config(plot_config_path)
        # set all parameters based on dict
        for k, v in config_params.items():
            CONFIG.set_value(k, v)
        for k, v in config_plot_params.items():
            CONFIG.set_value(k, v)
        CONFIG.set_value('val_batch_size',int(words_count))
        val_SNRs = np.linspace(config_plot_params['val_SNR_start'], config_plot_params['val_SNR_end'], num=config_plot_params['val_num_SNR'])
        dec = PolarFGTrainer()
        run_name = f"{graph_params['label']} crc dist-{type} counts-{int(words_count)}"
        crc = self.get_crc_plot(dec, run_name, type=type, only_crc_errors=only_crc_errors)
        bins = 16
        if 'bins' in graph_params.keys():
            bins = graph_params['bins']
        max_val = 2**CONFIG.crc_order
        step = int(max_val/bins)
        shift = 1 if only_crc_errors else 0
        crc_vals = np.arange(start=shift, stop=max_val+1, step=step)
        xlabel = "crc value"
        if type == 'uniform4':
            step = 1
            b = max_val/4
            crc_vals = [f'{int(i*b)}-{int((i+1)*b)}'for i in range(4)]
            max_val = 3
            xlabel = 'crc range'
        elif type == 'sum':
            step = 1
            max_val = CONFIG.crc_order
            crc_vals = np.arange(start=0, stop=max_val+1, step=step)
            xlabel = 'crc bit sum'
        elif type == 'sum%4':
            step = 1
            max_val = 3
            crc_vals = np.arange(start=0, stop=max_val+1, step=step)
            xlabel = 'crc bit sum % 4'

        for j,snr in enumerate(val_SNRs):
            plt.figure()
            crc_tmp = crc[j]
            # crc_vals = crc_vals.reshape(np.shape(crc_tmp))
            crc_counts = np.zeros(np.shape(crc_vals))
            if step != 1: # quantize bins
                width = (max_val/bins)*0.8
                ptr = 0
                for idx in range(len(crc_vals)-1):
                    while(ptr < crc_vals[idx+1]):
                        crc_counts[idx] += crc_tmp[ptr]
                        ptr +=1
                        if ptr >= len(crc_tmp):
                            print("ERROR - out of range in quantization")
                            raise OverflowError
                            exit(-1)
                crc_counts[-1] = sum(crc_tmp[ptr:])
            else:
                width = 0.5
                crc_counts = crc_tmp
            title = f'CRC distribution - {type} @ snr: {snr}'
            if not type:
                title = f'CRC distribution @ snr: {snr}'
            plt.bar(crc_vals, crc_counts, width=width)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel('counts')
            plt.legend(loc='lower left', prop={'size': 15})

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


    ''' flops '''
    '''64 32'''
    graph,conf = get_polar_64_32()
    # f = get_flops_num(*get_polar_64_32(),num_of_decoders=4)
    # plot_avg_flops(ens_flops=f, ens_dec_num=4, config_params=conf, title="")
    # f = get_flops_num(*get_polar_64_32(),num_of_decoders=2)
    # plot_avg_flops(ens_flops=f, ens_dec_num=2, config_params=conf, title="")

    '''256 128'''
    graph,conf = get_polar_256_128()
    # f = get_flops_num(*get_polar_256_128(),num_of_decoders=4)
    # plot_avg_flops(ens_flops=f, ens_dec_num=4, config_params=conf, title="")

    # f = get_flops_num(*get_polar_256_128(),num_of_decoders=8)
    # plot_avg_flops(ens_flops=f, ens_dec_num=8, config_params=conf, title="")

    '''512 256'''
    graph,conf = get_polar_512_256()
    # f = get_flops_num(*get_polar_512_256(),num_of_decoders=4)
    # plot_avg_flops(ens_flops=f, ens_dec_num=4, config_params=conf, title="")

    # f = get_flops_num(*get_polar_512_256(),num_of_decoders=8)
    # plot_avg_flops(ens_flops=f, ens_dec_num=8, config_params=conf, title="")

    '''1024 512'''
    graph,conf = get_polar_1024_512()
    # f = get_flops_num(*get_polar_1024_512(),num_of_decoders=4)
    # plot_avg_flops(ens_flops=f, ens_dec_num=4, config_params=conf, title="")

    # f = get_flops_num(*get_polar_1024_512(),num_of_decoders=8)
    # plot_avg_flops(ens_flops=f, ens_dec_num=8, config_params=conf, title="")

    ''' CRC '''
    plotter = Plotter(run_over=True, type='CRCPASS')
    plotter.plot(*get_ensemble_polar_64_32_crc11_iter5_decs_4_uniform(),dec_type='Ensemble')

    ''' CRC dist '''
    # words_count = 1e5
    # plotter = Plotter(run_over=False, type='pred_crc')
    # plotter.plot_crc(*get_polar_64_32(), type='', only_crc_errors=True, words_count=words_count)
    # plotter = Plotter(run_over=False, type='pred_crc')
    # plotter.plot_crc(*get_polar_64_32(), type='uniform4', only_crc_errors=True, words_count=words_count)
    # plotter = Plotter(run_over=False, type='pred_crc')
    # plotter.plot_crc(*get_polar_64_32(), type='sum', only_crc_errors=True, words_count=words_count)
    # plotter = Plotter(run_over=False, type='pred_crc')
    # plotter.plot_crc(*get_polar_64_32(), type='sum%4', only_crc_errors=True, words_count=words_count)


    ''' 64 32 '''
    # plotter = Plotter(run_over=True, type='BER')
    # plotter.plot(*get_polar_64_32(),dec_type='FG')
    # plotter.plot(*get_weighted_polar_64_32_iter5_crc11(),dec_type='FG')
    # plotter.plot(*get_ensemble_polar_64_32_crc11_iter5_decs_4_uniform(),dec_type='Ensemble')
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





# path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, 'figure.png'), bbox_inches='tight')
    plt.show()
