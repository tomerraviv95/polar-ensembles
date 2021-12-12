from python_code.plotter.plotter import *
from python_code.trainers.fg_trainer import PolarFGTrainer
from python_code.trainers.ensemble_trainer import EnsembleTrainer
from dir_definitions import PLOTS_DIR, FIGURES_DIR, WEIGHTS_DIR
import matplotlib.pyplot as plt
from globals import CONFIG
import datetime
import os
from time import time



def get_polar_64_32():
    graph_params = {'color': 'red', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 64, 'info_len': 32, 'load_weights': False , 'iteration_num':9, 'crc_order': 0}
    script_params = {"decoder_type":"FG"}
    return graph_params, runs_params, script_params


def get_weighted_polar_64_32_crc11():
    graph_params = {'color': 'green', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 64, 'info_len': 32, 'load_weights': True , 'iteration_num':9, 'crc_order': 11}
    script_params = {"decoder_type":"WFG"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_64_32_crc11():
    graph_params = {'color': 'blue', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 64, 'info_len': 32, 'load_weights': True , 'iteration_num':9, 'crc_order': 11}
    script_params = {"decoder_type":"Ensemble"}
    return graph_params, runs_params, script_params

def get_polar_256_128():
    graph_params = {'color': 'red', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': False , 'iteration_num':6, 'crc_order': 11}
    script_params = {"decoder_type":"FG"}
    return graph_params, runs_params, script_params

def get_weighted_polar_256_128_crc11_iter6():
    graph_params = {'color': 'green', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128,'run_name':'wfg_256_128_crc11_iter6', 'load_weights': True , 'iteration_num':6, 'crc_order': 11}
    script_params = {"decoder_type":"WFG"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_256_128_crc11_iter6():
    graph_params = {'color': 'blue', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128,'run_name':'ensemble_256_128_crc11_iter6', 'load_weights': True , 'iteration_num':6, 'crc_order': 11}
    script_params = {"decoder_type":"Ensemble"}
    return graph_params, runs_params, script_params

def TrainDecs(decs_to_train):
    trained = []
    # Train the decoders
    for dec in decs_to_train:
        graph_params, runs_params, script_params = dec()
        runs_params['load_weights'] = False
        for k, v in runs_params.items():
            CONFIG.set_value(k, v)

        if 'run_name' not in runs_params:
            run_name = f"{script_params['decoder_type']}_{runs_params['code_len']}_{runs_params['info_len']}_iters{runs_params['iteration_num']}_crc{runs_params['crc_order']}"
            CONFIG.set_value('run_name', run_name)
            runs_params['run_name'] = run_name
            print(f"no run name given, setting one auto : {run_name}")

        if 'label' not in graph_params:
            label = f"{script_params['decoder_type']} ({runs_params['code_len']},{runs_params['info_len']}) iters {runs_params['iteration_num']}"
            graph_params['label'] = label

        if script_params["decoder_type"] is "Ensemble":
            dec = EnsembleTrainer()
        else:
            dec = PolarFGTrainer()

        start = time()
        ber, fer = dec.train()
        end = time()
        script_params['train_time'] = end - start
        print(f"############ run named: {runs_params['run_name']} took {script_params['train_time']} sec to train ############")
        runs_params['load_weights'] = True
        params = {'graph_params':graph_params, 'runs_params':runs_params, 'script_params':script_params}
        trained.append(params)

    return trained

def PlotDecs(decs_to_plot, decs_to_calc_and_plot, dec_trained_params, plot_type="FER"):
    ''' Plot the decoders: decs_to_plot will only load its plots'''

    # these only needs to be plotted
    plotter = Plotter(run_over=False, type=plot_type)
    for dec in decs_to_plot:
        graph_params, runs_params, script_params = dec()
        if 'label' not in graph_params:
            label = f"{script_params['decoder_type']} ({runs_params['code_len']},{runs_params['info_len']})"
        plotter.plot(graph_params, runs_params, dec_type=script_params['decoder_type'])

    plotter = Plotter(run_over=True, type=plot_type)
    for dec in decs_to_calc_and_plot:
        graph_params, runs_params, script_params = dec()
        if 'label' not in graph_params:
            label = f"{script_params['decoder_type']} ({runs_params['code_len']},{runs_params['info_len']}) iters {runs_params['iteration_num']}"
            graph_params['label'] = label
        plotter.plot(graph_params, runs_params, dec_type=script_params['decoder_type'])

    # these needs to be evaluated at SNR's after training
    for dec in dec_trained_params:
        graph_params = dec['graph_params']
        runs_params = dec['runs_params']
        script_params = dec['script_params']
        plotter.plot(graph_params, runs_params, dec_type=script_params['decoder_type'])


    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, 'figure.png'), bbox_inches='tight')
    plt.show()


decoder_type = {"Ensemble":EnsembleTrainer, "FG":PolarFGTrainer}

if __name__ == '__main__':
    decs_to_train_and_plot = [get_weighted_polar_64_32_crc11, get_ensemble_polar_64_32_crc11] #[get_polar_64_32, get_weighted_polar_64_32_crc11_iter6] # decoder to train
    decs_to_plot_only = [get_polar_64_32] #[get_ensemble_polar_256_128_crc11_iter6, get_weighted_polar_256_128_crc11_iter6, get_polar_256_128] # decoders only plot
    decs_to_load_plot = [] # decoders only to load existing plot

    dec_trained_params = TrainDecs(decs_to_train_and_plot)

    PlotDecs(decs_to_load_plot, decs_to_plot_only, dec_trained_params, plot_type="FER")

