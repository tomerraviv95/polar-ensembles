from python_code.plotter.plotter import Plotter
from python_code.trainers.fg_trainer import PolarFGTrainer
from python_code.trainers.ensemble_trainer import EnsembleTrainer
from dir_definitions import PLOTS_DIR, FIGURES_DIR, WEIGHTS_DIR
import matplotlib.pyplot as plt
from globals import CONFIG, DEVICE
from python_code.utils.color_prints import *
import datetime
import os
from time import time



def get_polar_64_32_iter5():
    graph_params = {'color': 'black', 'marker': 'o', 'label':'BP (64,32) 5 iterations'}
    runs_params = {'code_type': 'Polar', 'code_len': 64, 'info_len': 32, 'load_weights': False, 'iteration_num':5, 'crc_order': 11}
    script_params = {"decoder_type":"FG"}
    return graph_params, runs_params, script_params


def get_weighted_polar_64_32_crc11_iter5():
    graph_params = {'color': 'red', 'marker': 'x', 'label':'WBP (64,32) 5 iterations'}
    runs_params = {'code_type': 'Polar', 'code_len': 64, 'info_len': 32, 'load_weights': True , 'num_of_epochs':200, 'iteration_num':5, 'crc_order': 11}
    script_params = {"decoder_type": "WFG"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_64_32_crc11_iter5_decs_2():
    graph_params = {'color': 'blue', 'marker': '*', 'label':'Ensemble (64,32) 5 iterations 2 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 64, 'info_len': 32, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':2, 'ensemble_crc_dist':'uniform', 'test_errors':100, 'train_minibatch_size':1000}
    script_params = {"decoder_type": "Ensemble"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_64_32_crc11_iter5_decs_4():
    graph_params = {'color': 'green', 'marker': '*', 'label':'Ensemble (64,32) 5 iterations 4 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 64, 'info_len': 32, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':4, 'ensemble_crc_dist':'uniform', 'test_errors':100, 'train_minibatch_size':2000}
    script_params = {"decoder_type": "Ensemble"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_64_32_crc11_iter5_decs_8():
    graph_params = {'color': 'magenta', 'marker': '*', 'label':'Ensemble (64,32) 5 iterations 8 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 64, 'info_len': 32, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':8, 'ensemble_crc_dist':'uniform', 'test_errors':100, 'train_minibatch_size':4000}
    script_params = {"decoder_type": "Ensemble"}
    return graph_params, runs_params, script_params
##################################
def get_polar_256_128_crc11_iter10():
    graph_params = {'color': 'black', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': False, 'iteration_num':10, 'crc_order': 11}
    script_params = {"decoder_type":"BP"}
    return graph_params, runs_params, script_params


def get_weighted_polar_256_128_crc11_iter10():
    graph_params = {'color': 'red', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True , 'num_of_epochs':200, 'iteration_num':10, 'crc_order': 11, 'train_minibatch_size': 500}
    script_params = {"decoder_type":"WBP"}
    return graph_params, runs_params, script_params

def get_weighted_polar_256_128_crc11_iter40():
    graph_params = {'color': 'green', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True , 'num_of_epochs':200, 'iteration_num':40, 'crc_order': 11, 'train_minibatch_size': 500}
    script_params = {"decoder_type":"WBP"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_256_128_crc11_iter10_decs_4():
    graph_params = {'color': 'green', 'marker': '*', 'label':'Ensemble (256,128) 10 iterations 4 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 10, 'crc_order': 11, 'ensemble_dec_num':4, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

def get_ensemble_polar_64_32_crc11_iter5_decs_2_best():
    graph_params = {'color': 'blue', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 64, 'info_len': 32, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':2, 'ensemble_crc_dist':'uniform', 'test_errors':100, 'train_minibatch_size':2000}
    script_params = {"decoder_type": "Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

##################################

##################################
def get_polar_256_128_crc11_iter7():
    graph_params = {'color': 'black', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': False, 'iteration_num':7, 'crc_order': 11}
    script_params = {"decoder_type":"BP"}
    return graph_params, runs_params, script_params


def get_weighted_polar_256_128_crc11_iter7():
    graph_params = {'color': 'red', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True , 'num_of_epochs':200, 'iteration_num':7, 'crc_order': 11, 'train_minibatch_size': 500}
    script_params = {"decoder_type":"WBP"}
    return graph_params, runs_params, script_params

def get_weighted_polar_256_128_crc11_iter28():
    graph_params = {'color': 'green', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True , 'num_of_epochs':200, 'iteration_num':28, 'crc_order': 11, 'train_minibatch_size': 500}
    script_params = {"decoder_type":"WBP"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_256_128_crc11_iter7_decs_4():
    graph_params = {'color': 'green', 'marker': '*', 'label':'Ensemble (256,128) 5 iterations 4 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 7, 'crc_order': 11, 'ensemble_dec_num':4, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

########################################
def get_polar_128_64_crc11_iter5():
    graph_params = {'color': 'black', 'marker': 'o', 'label':'BP (128,64) 5 iterations'}
    runs_params = {'code_type': 'Polar', 'code_len': 128, 'info_len': 64, 'load_weights': False, 'iteration_num':5, 'crc_order': 11}
    script_params = {"decoder_type":"BP"}
    return graph_params, runs_params, script_params

def get_weighted_polar_128_64_crc11_iter5():
    graph_params = {'color': 'red', 'marker': 'o', 'label':'WBP (128,64) 5 iterations'}
    runs_params = {'code_type': 'Polar', 'code_len': 128, 'info_len': 64, 'load_weights': True , 'num_of_epochs':200, 'iteration_num':5, 'crc_order': 11, 'train_minibatch_size': 500}
    script_params = {"decoder_type":"WBP"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_128_64_crc11_iter5_decs_2_best():
    graph_params = {'color': 'blue', 'marker': 'X', 'label':'Ensemble (128,64) 5 iterations 2 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 128, 'info_len': 64, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':2, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

def get_ensemble_polar_128_64_crc11_iter5_decs_4_best():
    graph_params = {'color': 'green', 'marker': '*', 'label':'Ensemble (128,64) 5 iterations 4 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 128, 'info_len': 64, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':4, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

def get_ensemble_polar_128_64_crc11_iter5_decs_8_best():
    graph_params = {'color': 'magenta', 'marker': '*', 'label':'Ensemble (128,64) 5 iterations 8 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 128, 'info_len': 64, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':8, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':3500}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

##################################
def get_polar_256_128_crc11_iter5():
    graph_params = {'color': 'black', 'marker': 'o', 'label':'BP (256,128) 5 iterations'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': False, 'iteration_num':5, 'crc_order': 11}
    script_params = {"decoder_type":"BP"}
    return graph_params, runs_params, script_params

def get_weighted_polar_256_128_crc11_iter10():
    graph_params = {'color': 'blue', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True , 'num_of_epochs':100, 'iteration_num':10, 'crc_order': 11, 'train_minibatch_size': 500}
    script_params = {"decoder_type":"WBP"}
    return graph_params, runs_params, script_params

def get_weighted_polar_256_128_crc11_iter20():
    graph_params = {'color': 'green', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True , 'num_of_epochs':100, 'iteration_num':20, 'crc_order': 11, 'train_minibatch_size': 500}
    script_params = {"decoder_type":"WBP"}
    return graph_params, runs_params, script_params

def get_weighted_polar_256_128_crc11_iter40():
    graph_params = {'color': 'magenta', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True , 'num_of_epochs':100, 'iteration_num':40, 'crc_order': 11, 'train_minibatch_size': 500}
    script_params = {"decoder_type":"WBP"}
    return graph_params, runs_params, script_params

def get_weighted_polar_256_128_crc11_iter5():
    graph_params = {'color': 'red', 'marker': 'o', 'label':'WBP (256,128) 5 iterations'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True , 'num_of_epochs':100, 'iteration_num':5, 'crc_order': 11, 'train_minibatch_size': 500}
    script_params = {"decoder_type":"WBP"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_256_128_crc11_iter5_decs_2():
    graph_params = {'color': 'blue', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True, 'num_of_epochs':100, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':2, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_256_128_crc11_iter5_decs_4():
    graph_params = {'color': 'blue', 'marker': 'x', 'label':'Ensemble1 (256,128) 5 iterations 4 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True, 'num_of_epochs':100, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':4, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_256_128_crc11_iter5_decs_6():
    graph_params = {'color': 'magenta', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True, 'num_of_epochs':100, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':6, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_256_128_crc11_iter5_decs_8():
    graph_params = {'color': 'magenta', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True, 'num_of_epochs':100, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':8, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':3500}
    script_params = {"decoder_type":"Ensemble"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_256_128_crc11_iter5_decs_2_best():
    graph_params = {'color': 'blue', 'marker': 'X', 'label':'Ensemble (256,128) 5 iterations 2 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True, 'num_of_epochs':100, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':2, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

def get_ensemble_polar_256_128_crc11_iter5_decs_4_best():
    graph_params = {'color': 'green', 'marker': '*', 'label':'Ensemble (256,128) 5 iterations 4 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True, 'num_of_epochs':100, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':4, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

def get_ensemble_polar_256_128_crc11_iter5_decs_6_best():
    graph_params = {'color': 'magenta', 'marker': '*'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True, 'num_of_epochs':100, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':6, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

def get_ensemble_polar_256_128_crc11_iter5_decs_8_best():
    graph_params = {'color': 'magenta', 'marker': '*', 'label':'Ensemble (256,128) 5 iterations 8 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 256, 'info_len': 128, 'load_weights': True, 'num_of_epochs':100, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':8, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':3500}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

########################################
def get_polar_512_256_crc11_iter5():
    graph_params = {'color': 'black', 'marker': 'o', 'label':'BP (512,256) 5 iterations'}
    runs_params = {'code_type': 'Polar', 'code_len': 512, 'info_len': 256, 'load_weights': False, 'iteration_num':5, 'crc_order': 11}
    script_params = {"decoder_type":"BP"}
    return graph_params, runs_params, script_params

def get_weighted_polar_512_256_crc11_iter5():
    graph_params = {'color': 'red', 'marker': 'o', 'label':'WBP (512,256) 5 iterations'}
    runs_params = {'code_type': 'Polar', 'code_len': 512, 'info_len': 256, 'load_weights': True , 'num_of_epochs':200, 'iteration_num':5, 'crc_order': 11, 'train_minibatch_size': 500}
    script_params = {"decoder_type":"WBP"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_512_256_crc11_iter5_decs_2_best():
    graph_params = {'color': 'blue', 'marker': 'X', 'label':'Ensemble (512,256) 5 iterations 2 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 512, 'info_len': 256, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':2, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

def get_ensemble_polar_512_256_crc11_iter5_decs_4_best():
    graph_params = {'color': 'green', 'marker': '*', 'label':'Ensemble (512,256) 5 iterations 4 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 512, 'info_len': 256, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':4, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

def get_ensemble_polar_512_256_crc11_iter5_decs_8_best():
    graph_params = {'color': 'magenta', 'marker': '*', 'label':'Ensemble (512,256) 5 iterations 8 decoders'}
    runs_params = {'code_type': 'Polar', 'code_len': 512, 'info_len': 256, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':8, 'ensemble_crc_dist':'uniform', 'test_errors':500, 'train_minibatch_size':3500}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params



##################################

def get_polar_1024_512_crc11():
    graph_params = {'color': 'black', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 1024, 'info_len': 512, 'load_weights': False, 'iteration_num':5, 'crc_order': 11}
    script_params = {"decoder_type":"FG"}
    return graph_params, runs_params, script_params

def get_weighted_polar_1024_512_crc11():
    graph_params = {'color': 'red', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 1024, 'info_len': 512, 'load_weights': True , 'num_of_epochs':200, 'iteration_num':5, 'crc_order': 11, 'train_minibatch_size': 500}
    script_params = {"decoder_type":"WFG"}
    return graph_params, runs_params, script_params

def get_weighted_polar_1024_512_crc11_iter20():
    graph_params = {'color': 'red', 'marker': 'o'}
    runs_params = {'code_type': 'Polar', 'code_len': 1024, 'info_len': 512, 'load_weights': True , 'num_of_epochs':200, 'iteration_num':20, 'crc_order': 11, 'train_minibatch_size': 500}
    script_params = {"decoder_type":"WFG"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_1024_512_crc11_iter5_decs_2_uniform():
    graph_params = {'color': 'blue', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 1024, 'info_len': 512, 'load_weight': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':2, 'ensemble_crc_dist':'uniform', 'test_errors':100, 'train_minibatch_size':1000}
    script_params = {"decoder_type":"Ensemble"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_1024_512_crc11_iter5_decs_4_uniform():
    graph_params = {'color': 'green', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 1024, 'info_len': 512, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':4, 'ensemble_crc_dist':'uniform', 'test_errors':100, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_1024_512_crc11_iter5_decs_8_uniform():
    graph_params = {'color': 'magenta', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 1024, 'info_len': 512, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':8, 'ensemble_crc_dist':'uniform', 'test_errors':100, 'train_minibatch_size':3000}
    script_params = {"decoder_type":"Ensemble"}
    return graph_params, runs_params, script_params

def get_ensemble_polar_1024_512_crc11_iter5_decs_2_uniform_best():
    graph_params = {'color': 'blue', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 1024, 'info_len': 512, 'load_weight': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':2, 'ensemble_crc_dist':'uniform', 'test_errors':100, 'train_minibatch_size':1000}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

def get_ensemble_polar_1024_512_crc11_iter5_decs_4_uniform_best():
    graph_params = {'color': 'green', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 1024, 'info_len': 512, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':4, 'ensemble_crc_dist':'uniform', 'test_errors':100, 'train_minibatch_size':2000}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params

def get_ensemble_polar_1024_512_crc11_iter5_decs_8_uniform_best():
    graph_params = {'color': 'magenta', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 1024, 'info_len': 512, 'load_weights': True, 'num_of_epochs':200, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':8, 'ensemble_crc_dist':'uniform', 'test_errors':100, 'train_minibatch_size':4000}
    script_params = {"decoder_type":"Ensemble", 'take_crc_0':True}
    return graph_params, runs_params, script_params
###################################

def updateParams(graph_params, runs_params, script_params):
    script_params['take_crc_0'] = takeBestDec(graph_params, runs_params, script_params)
    runs_params['run_name'] = getRunName(graph_params, runs_params, script_params)
    graph_params['label'] = getLabel(graph_params, runs_params, script_params)

def takeBestDec(graph_params, runs_params, script_params):
    if 'take_crc_0' in script_params.keys():
        return script_params['take_crc_0']
    return False

def getRunName(graph_params, runs_params, script_params):
    if 'run_name' in runs_params.keys():
        if runs_params['run_name']:
            return runs_params['run_name']

    if script_params["decoder_type"] is "Ensemble":
        run_name = f"ensemble_{runs_params['code_len']}_{runs_params['info_len']}_iters{runs_params['iteration_num']}_crc{runs_params['crc_order']}_{runs_params['ensemble_crc_dist']}_decs_{runs_params['ensemble_dec_num']}"
    else:
        run_name = f"wfg_{runs_params['code_len']}_{runs_params['info_len']}_iters{runs_params['iteration_num']}_crc{runs_params['crc_order']}"
    LOGD(f" run name: {run_name}")
    return run_name

def getLabel(graph_params, runs_params, script_params):
    if 'label' in graph_params:
        return graph_params['label']
    label = f"{script_params['decoder_type']} ({runs_params['code_len']},{runs_params['info_len']}) iters {runs_params['iteration_num']} crc{runs_params['crc_order']}"
    if script_params["decoder_type"] is "Ensemble":
        label += f" {runs_params['ensemble_crc_dist']} decs {runs_params['ensemble_dec_num']}"
        if not script_params['take_crc_0']:
            label += f" regular"
    return label

def TrainDecs(decs_to_train):
    trained = []
    # CONFIG.set_value('run_name', '')
    # Train the decoders
    for dec_name in decs_to_train:
        CONFIG.load_default_config()
        graph_params, runs_params, script_params = dec_name()
        runs_params['load_weights'] = False
        for k, v in runs_params.items():
            CONFIG.set_value(k, v)

        if script_params["decoder_type"] is "Ensemble":
            dec = EnsembleTrainer()
        else:
            dec = PolarFGTrainer()

        start = time()
        ber, fer = dec.train()
        end = time()
        script_params['train_time'] = end - start
        dir = dec.weights_dir
        run_name = dir.split("\\")[-1]
        print(f"############ run named: {run_name} took {round(script_params['train_time'])} sec to train ############")
        runs_params['load_weights'] = True
        if 'run_name' not in runs_params.keys():
            runs_params['run_name'] = run_name
        LOGD(run_name)
        # if 'label' not in graph_params:
        #     label = run_name # f"{script_params['decoder_type']} ({runs_params['code_len']},{runs_params['info_len']}) iters {runs_params['iteration_num']}"
        #     graph_params['label'] = label

        params = {'graph_params':graph_params, 'runs_params':runs_params, 'script_params':script_params}
        trained.append(params)

        CONFIG.load_default_config()

    return trained

def PlotDecs(decs_to_plot, decs_to_calc_and_plot, dec_trained_params, plot_type="FER"):
    ''' Plot the decoders: decs_to_plot will only load its plots'''

    # these only needs to be plotted
    plotter = Plotter(run_over=False, type=plot_type)
    for dec_params_func in decs_to_plot:
        graph_params, runs_params, script_params = dec_params_func()
        updateParams(graph_params, runs_params, script_params)
        plotter.plot(graph_params, runs_params, dec_type=script_params['decoder_type'])

    plotter = Plotter(run_over=True, type=plot_type)
    for dec_params_func in decs_to_calc_and_plot:
        graph_params, runs_params, script_params = dec_params_func()
        updateParams(graph_params, runs_params, script_params)
        start = time()
        plotter.plot(graph_params, runs_params, dec_type=script_params['decoder_type'], take_crc_0=script_params['take_crc_0'])
        end = time()
        print(f"############ graph named: {graph_params['label']} took {round(end-start)} sec to plot ############")

    # these needs to be evaluated at SNR's after training
    for dec_params in dec_trained_params:
        graph_params = dec_params['graph_params']
        runs_params = dec_params['runs_params']
        script_params = dec_params['script_params']
        updateParams(graph_params, runs_params, script_params)
        start = time()
        plotter.plot(graph_params, runs_params, dec_type=script_params['decoder_type'], take_crc_0=script_params['take_crc_0'])
        end = time()
        print(f"############ graph named: {graph_params['label']} took {round(end-start)} sec to plot ############")

    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, 'figure.png'), bbox_inches='tight')
    plt.show()


decoder_type = {"Ensemble":EnsembleTrainer, "FG":PolarFGTrainer}

if __name__ == '__main__':
    '''
    decs_to_train_and_plot - will train and plot them
    decs_to_plot_only - will plot them
    decs_to_load_plot - will load existing plot
    '''
    decs_to_train_and_plot = []
    decs_to_plot_only = []
    decs_to_load_plot = []

    decs_to_plot_only = [get_polar_64_32_iter5]
    decs_to_load_plot = []
    decs_to_train_and_plot = [get_weighted_polar_64_32_crc11_iter5, get_ensemble_polar_64_32_crc11_iter5_decs_2, get_ensemble_polar_64_32_crc11_iter5_decs_4,get_ensemble_polar_64_32_crc11_iter5_decs_8]

    # decs_to_train_and_plot = []
    # decs_to_plot_only = [get_weighted_polar_1024_512_crc11_iter20 ,get_ensemble_polar_1024_512_crc11_iter5_decs_2_uniform, get_ensemble_polar_1024_512_crc11_iter5_decs_2_uniform_best, get_ensemble_polar_1024_512_crc11_iter5_decs_8_uniform_best]
    # decs_to_load_plot = [get_polar_1024_512_crc11, get_weighted_polar_1024_512_crc11, get_ensemble_polar_1024_512_crc11_iter5_decs_4_uniform, get_ensemble_polar_1024_512_crc11_iter5_decs_4_uniform_best]
    # decs_to_load_plot += []

    # decs_to_train_and_plot = [get_weighted_polar_256_128_crc11_iter3, get_weighted_polar_256_128_crc11_iter12 , get_ensemble_polar_256_128_crc11_iter3_decs_4]
    # decs_to_plot_only = [get_polar_256_128_crc11_iter3]
    # decs_to_load_plot = []

    # decs_to_train_and_plot = []
    # decs_to_plot_only = []
    # decs_to_load_plot = [get_polar_256_128_crc11_iter10, get_weighted_polar_256_128_crc11_iter10, get_weighted_polar_256_128_crc11_iter40, get_ensemble_polar_256_128_crc11_iter10_decs_4]

    # decs_to_load_plot = [get_polar_256_128_crc11, get_weighted_polar_256_128_crc11, get_weighted_polar_256_128_crc11_iter10, get_weighted_polar_256_128_crc11_iter20, get_weighted_polar_256_128_crc11_iter40, get_ensemble_polar_256_128_crc11_iter5_decs_2_best, get_ensemble_polar_256_128_crc11_iter5_decs_4_best, get_ensemble_polar_256_128_crc11_iter5_decs_8_best]
    #
    # decs_to_load_plot = [get_polar_256_128_crc11, get_weighted_polar_256_128_crc11, get_weighted_polar_256_128_crc11_iter10, get_ensemble_polar_256_128_crc11_iter5_decs_2_best]


    # decs_to_plot_only = [get_polar_256_128_crc11_iter5, get_weighted_polar_256_128_crc11_iter5, get_ensemble_polar_256_128_crc11_iter5_decs_2_best, get_ensemble_polar_256_128_crc11_iter5_decs_4_best]
    # decs_to_train_and_plot = [get_ensemble_polar_256_128_crc11_iter5_decs_8_best]
    # decs_to_load_plot = []

    # decs_to_train_and_plot = []
    # decs_to_load_plot = [get_polar_128_64_crc11_iter5, get_weighted_polar_128_64_crc11_iter5, get_ensemble_polar_128_64_crc11_iter5_decs_2_best, get_ensemble_polar_128_64_crc11_iter5_decs_4_best,get_ensemble_polar_128_64_crc11_iter5_decs_8_best]
    # decs_to_load_plot = [get_polar_512_256_crc11_iter5, get_weighted_polar_512_256_crc11_iter5, get_ensemble_polar_512_256_crc11_iter5_decs_2_best, get_ensemble_polar_512_256_crc11_iter5_decs_4_best,get_ensemble_polar_512_256_crc11_iter5_decs_8_best]


    dec_trained_params = TrainDecs(decs_to_train_and_plot)

    PlotDecs(decs_to_load_plot, decs_to_plot_only, dec_trained_params, plot_type="FER")


