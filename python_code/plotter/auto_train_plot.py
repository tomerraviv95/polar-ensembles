from python_code.plotter.plotter import *
from python_code.trainers.fg_trainer import PolarFGTrainer
from python_code.trainers.ensemble_trainer import EnsembleTrainer
from dir_definitions import PLOTS_DIR, FIGURES_DIR, WEIGHTS_DIR
import matplotlib.pyplot as plt
from globals import CONFIG
import datetime
import os



def get_polar_64_32():
    graph_params = {'label': 'FG (64,32)', 'color': 'red', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 64, 'info_len': 32, 'num_of_epochs':5, 'iteration_num':5, 'crc_order': 0}
    script_params = {"decoder_type":"FG"}
    return graph_params, runs_params, script_params


def get_weighted_polar_64_32_crc11_iter6():
    graph_params = {'label': 'WFG (64,32) 6 iters crc order 11', 'color': 'red', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 64, 'info_len': 32, 'num_of_epochs':5, 'iteration_num':5, 'crc_order': 11}
    script_params = {"decoder_type":"FG"}
    return graph_params, runs_params, script_params


def TrainDecs(decs_to_train):
    trained = []
    # Train the decoders
    for dec in decs_to_train:
        graph_params, runs_params, script_params = dec()
        for k, v in runs_params.items():
            CONFIG.set_value(k, v)

        runs_params['load_weights'] = True

        if 'run_name' not in runs_params:
            run_name = f"{runs_params['code_type']}_{script_params['decoder_type']}_{runs_params['code_len']}_{runs_params['info_len']}_iters{runs_params['iteration_num']}_crc{runs_params['crc_order']}"
            CONFIG.set_value('run_name', run_name)
            runs_params['run_name'] = run_name
            print(f"no run name given, setting one auto : {run_name}")

        params = {'graph_params':graph_params, 'runs_params':runs_params, 'script_params':script_params}
        trained.append(params)

        if script_params["decoder_type"] is "Ensemble":
            dec = EnsembleTrainer()
        else:
            dec = PolarFGTrainer()

        ber, fer = dec.train()

    return trained

def PlotDecs(decs_to_plot, decs_to_calc_and_plot, plot_type="FER"):
    ''' Plot the decoders: decs_to_plot will only load its plots'''

    # these only needs to be plotted
    plotter = Plotter(run_over=False, type=plot_type)
    for dec in decs_to_plot:
        graph_params, runs_params, script_params = dec()
        plotter.plot(graph_params, runs_params, dec_type=script_params['decoder_type'])

    # these needs to be evaluated at SNR's after training
    plotter = Plotter(run_over=True, type=plot_type)
    for dec in decs_to_calc_and_plot:
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
    decs_to_train = [get_polar_64_32, get_weighted_polar_64_32_crc11_iter6] # decoder to train
    decs_to_plot_with_decs_to_train = [] # decoders only to plot

    dec_trained_params = TrainDecs(decs_to_train)

    PlotDecs(decs_to_plot_with_decs_to_train, dec_trained_params)

