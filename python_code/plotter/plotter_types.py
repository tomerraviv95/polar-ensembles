def get_polar_64_32():
    graph_params = {'label': 'BP (64,32) iters 5 crc11', 'color': 'black', 'marker': 'o'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'polar_fg_64_32', 'iteration_num': 5, 'load_weights': False}
    return graph_params, runs_params

def get_weighted_polar_64_32_iter5_crc11():
    graph_params = {'label': 'WFG (64,32) 5 iters crc order 11', 'color': 'red', 'marker': 'o'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'wfg_64_32_iters5_crc11', 'load_weights': True}
    return graph_params, runs_params

def get_new_ensemble_64_32_iters5_crc11_sum_decs_4():
    graph_params = {'label': 'new ensemble (64,32) 5 iters crc order 11', 'color': 'blue', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'new_ensemble_64_32_iters5_crc11_sum_decs_4', 'load_weights': True}
    return graph_params, runs_params

def get_new_ensemble_64_32_iters5_crc11_sum_decs_4_best_dec():
    graph_params = {'label': 'new ensemble (64,32) 5 iters crc order 11, best dec', 'color': 'green', 'marker': '*'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'new_ensemble_64_32_iters5_crc11_sum_decs_4', 'load_weights': True}
    return graph_params, runs_params

def get_weighted_polar_64_32_crc11_iter6():
    graph_params = {'label': 'WFG (64,32) 6 iters crc order 11', 'color': 'blue', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'wfg_64_32_crc11_iter6', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_64_32_crc11_iter6():
    graph_params = {'label': 'ensemble (64,32) 6 iters crc order 11', 'color': 'red', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'ensemble_64_32_crc11_iter6', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_64_32_crc11_iter6_best_dec():
    graph_params = {'label': 'ensemble (64,32) 6 iters crc order 11, best dec', 'color': 'green', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'ensemble_64_32_crc11_iter6', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_64_32_iter6_crc11_sum():
    graph_params = {'label': 'ensemble (64,32) 6 iters crc order 11 sum crc bits', 'color': 'green', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'ensemble_64_32_iters6_crc11_sum', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_64_32_iter6_crc11_sum_mod():
    graph_params = {'label': 'ensemble (64,32) 6 iters crc order 11 sum crc mod 4', 'color': 'green', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'ensemble_64_32_iters6_crc11_sum_mod', 'load_weights': True}
    return graph_params, runs_params

def get_weighted_polar_64_32_iter6_crc11():
    graph_params = {'label': 'WFG (64,32) 6 iters crc order 11', 'color': 'blue', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'WFG_64_32_iter6_crc11', 'load_weights': True}
    return graph_params, runs_params

def get_weighted_polar_64_32_crc11_iter30():
    graph_params = {'label': 'WFG (64,32) 6 iters crc order 11', 'color': 'blue', 'marker': 'o'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'wfg_64_32_crc11_iter30', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_64_32_iter6_crc11():
    graph_params = {'label': 'ensemble (64,32) 6 iters crc order 11', 'color': 'red', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'Ensemble_64_32_iter6_crc11', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_64_32_iter6_crc11_best_dec():
    graph_params = {'label': 'ensemble (64,32) 6 iters crc order 11, best dec', 'color': 'green', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'Ensemble_64_32_iter6_crc11', 'load_weights': True}
    return graph_params, runs_params

def get_weighted_polar_64_32_crc11_iter5():
    graph_params = {'label': 'WFG (64,32) 5 iters crc order 11', 'color': 'blue', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'wfg_64_32_crc11_iter5', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_64_32_crc11_iter5():
    graph_params = {'label': 'ensemble (64,32) 5 iters crc order 11', 'color': 'red', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'ensemble_64_32_crc11_iter5', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_64_32_crc11_iter5_best_dec():
    graph_params = {'label': 'ensemble (64,32) 5 iters crc order 11, best dec', 'color': 'green', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'ensemble_64_32_crc11_iter5', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_polar_64_32_crc11_iter5_decs_4_uniform():
    graph_params = {'color': 'blue', 'marker': 'x'}
    runs_params = {'code_type': 'Polar', 'code_len': 64, 'info_len': 32, 'run_name': 'ensemble_64_32_iters5_crc11_uniform_decs_4', 'load_weights': True, 'num_of_epochs':100, 'iteration_num': 5, 'crc_order': 11, 'ensemble_dec_num':4, 'ensemble_crc_dist':'uniform', 'test_errors':200, 'train_minibatch_size':2000}
    script_params = {"decoder_type": "Ensemble"}
    return graph_params, runs_params


def get_weighted_polar_64_32_iter5_crc11_snr_2_6():
    graph_params = {'label': 'WBP (64,32) 5 iterations', 'color': 'red', 'marker': 'o'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'wfg_64_32_iters5_crc11_snrs_2-6', 'load_weights': True}
    return graph_params, runs_params


def get_ensemble_64_32_iters5_crc11_uniform_decs_2_best_dec_snr_2_6():
    graph_params = {'label': 'Ensemble (64,32) 5 iterations 2 decoders', 'color': 'blue', 'marker': '*'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'ensemble_64_32_iters5_crc11_uniform_decs_2_snrs_2-6', 'load_weights': True}
    return graph_params, runs_params


def get_ensemble_64_32_iters5_crc11_uniform_decs_4_best_dec_snr_2_6():
    graph_params = {'label': 'Ensemble (64,32) 5 iterations 4 decoders', 'color': 'green', 'marker': '*'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'ensemble_64_32_iters5_crc11_uniform_decs_4_snrs_2-6', 'load_weights': True}
    return graph_params, runs_params


def get_ensemble_64_32_iters5_crc11_uniform_decs_8_best_dec_snr_2_6():
    graph_params = {'label': 'Ensemble (64,32) 5 iterations 8 decoders', 'color': 'magenta', 'marker': '*'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'ensemble_64_32_iters5_crc11_uniform_decs_8_snrs_2-6', 'load_weights': True}
    return graph_params, runs_params

def get_weighted_polar_128_64_iter5_crc11_snr_2_5():
    graph_params = {'label': 'WBP (128,64) 5 iterations', 'color': 'red', 'marker': 'o'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'wfg_128_64_iters5_crc11_snrs_2-5', 'load_weights': True}
    return graph_params, runs_params


def get_ensemble_128_64_iters5_crc11_uniform_decs_2_best_dec_snr_2_5():
    graph_params = {'label': 'Ensemble (128,64) 5 iterations 2 decoders', 'color': 'blue', 'marker': '*'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'ensemble_128_64_iters5_crc11_uniform_decs_2_snrs_2-5', 'load_weights': True}
    return graph_params, runs_params


def get_ensemble_128_64_iters5_crc11_uniform_decs_4_best_dec_snr_2_5():
    graph_params = {'label': 'Ensemble (128,64) 5 iterations 4 decoders', 'color': 'green', 'marker': '*'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'ensemble_128_64_iters5_crc11_uniform_decs_4_snrs_2-5', 'load_weights': True}
    return graph_params, runs_params


def get_ensemble_128_64_iters5_crc11_uniform_decs_8_best_dec_snr_2_5():
    graph_params = {'label': 'Ensemble (128,64) 5 iterations 8 decoders', 'color': 'magenta', 'marker': '*'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'ensemble_128_64_iters5_crc11_uniform_decs_8_snrs_2-5', 'load_weights': True}
    return graph_params, runs_params

def get_polar_128_64_iter5():
    graph_params = {'label': 'FG (128,64)', 'color': 'blue', 'marker': 'o'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'polar_fg_128_64', 'iteration_num': 5, 'load_weights': False}
    return graph_params, runs_params

def get_polar_128_64_iter6():
    graph_params = {'label': 'FG (128,64) 6 iterations', 'color': 'blue', 'marker': 'o'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'polar_fg_128_64', 'iteration_num': 6, 'load_weights': False}
    return graph_params, runs_params

def get_weighted_polar_128_64_iter6_crc11_snr_2_5():
    graph_params = {'label': 'WBP (128,64) 6 iterations', 'color': 'red', 'marker': 'o'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'wfg_128_64_iters6_crc11_snrs_2-5', 'load_weights': True}
    return graph_params, runs_params


def get_ensemble_128_64_iters6_crc11_uniform_decs_2_best_dec_snr_2_5():
    graph_params = {'label': 'Ensemble (128,64) 6 iterations 2 decoders', 'color': 'blue', 'marker': '*'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'ensemble_128_64_iters6_crc11_uniform_decs_2_snrs_2-5', 'load_weights': True}
    return graph_params, runs_params


def get_ensemble_128_64_iters6_crc11_uniform_decs_4_best_dec_snr_2_5():
    graph_params = {'label': 'Ensemble (128,64) 6 iterations 4 decoders', 'color': 'green', 'marker': '*'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'ensemble_128_64_iters6_crc11_uniform_decs_4_snrs_2-5', 'load_weights': True}
    return graph_params, runs_params


def get_ensemble_128_64_iters6_crc11_uniform_decs_8_best_dec_snr_2_5():
    graph_params = {'label': 'Ensemble (128,64) 6 iterations 8 decoders', 'color': 'magenta', 'marker': '*'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'ensemble_128_64_iters6_crc11_uniform_decs_8_snrs_2-5', 'load_weights': True}
    return graph_params, runs_params


def get_weighted_polar_128_64():
    graph_params = {'label': 'WFG (128,64)', 'color': 'blue', 'marker': 'x'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'polar_fg_128_64', 'load_weights': True}
    return graph_params, runs_params


def get_polar_256_128():
    graph_params = {'label': 'BP (256,128) iters 5 crc11', 'color': 'black', 'marker': 'o'}
    runs_params = {'code_len': 256, 'info_len': 128, 'run_name': 'polar_fg_256_128', 'iteration_num': 5, 'load_weights': False}
    return graph_params, runs_params


def get_weighted_polar_256_128():
    graph_params = {'label': 'WFG (256,128)', 'color': 'black', 'marker': 'x'}
    runs_params = {'code_len': 256, 'info_len': 128, 'run_name': 'wfg_256_128', 'load_weights': True}
    return graph_params, runs_params

def get_weighted_polar_256_128_crc11_iter6():
    graph_params = {'label': 'WFG (256,128) 6 iters crc order 11', 'color': 'blue', 'marker': 'x'}
    runs_params = {'code_len': 256, 'info_len': 128, 'run_name': 'wfg_256_128_crc11_iter6', 'load_weights': True}
    return graph_params, runs_params

def get_weighted_polar_256_128_crc11_iter30():
    graph_params = {'label': 'WFG (256,128) 30 iters crc order 11', 'color': 'blue', 'marker': 'o'}
    runs_params = {'code_len': 256, 'info_len': 128, 'run_name': 'wfg_256_128_crc11_iter30', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_256_128_crc11_iter6():
    graph_params = {'label': 'ensemble (256,128) 6 iters crc order 11', 'color': 'red', 'marker': 'x'}
    runs_params = {'code_len': 256, 'info_len': 128, 'run_name': 'ensemble_256_128_crc11_iter6', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_256_128_crc11_iter6_best_dec():
    graph_params = {'label': 'ensemble (256,128) 6 iters crc order 11, best dec', 'color': 'green', 'marker': '*'}
    runs_params = {'code_len': 256, 'info_len': 128, 'run_name': 'ensemble_256_128_crc11_iter6', 'load_weights': True}
    return graph_params, runs_params


def get_weighted_polar_256_128_iter7():
    graph_params = {'label': 'WFG (256,128) 7 iters', 'color': 'blue', 'marker': 'x'}
    runs_params = {'code_len': 256, 'info_len': 128, 'run_name': 'wfg_256_128_crc11_iter7', 'load_weights': True}
    return graph_params, runs_params


def get_weighted_polar_1024_512():
    graph_params = {'label': 'WFG (1024,512)', 'color': 'green', 'marker': 'x'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'run_name': 'wfg_1024_512_negative', 'load_weights': True}
    return graph_params, runs_params

def get_polar_1024_512():
    graph_params = {'label': 'FG (1024,512)', 'color': 'green', 'marker': 'o'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'run_name': 'polar_fg_1024_512', 'load_weights': False}
    return graph_params, runs_params


def get_weighted_polar_1024_512():
    graph_params = {'label': 'WFG (1024,512)', 'color': 'green', 'marker': 'x'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'run_name': 'wfg_1024_512_negative', 'load_weights': True}
    return graph_params, runs_params


def get_ensemble_64_32_test():
    graph_params = {'label': 'ensemble (64,32) 6 iters crc order 11', 'color': 'green', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'test', 'load_weights': True}
    return graph_params, runs_params

''' 512 256 '''
def get_polar_512_256():
    graph_params = {'label': 'BP (512,256) iters 5 crc11', 'color': 'black', 'marker': 'o'}
    runs_params = {'code_len': 512, 'info_len': 256, 'iteration_num': 5, 'load_weights': False}
    return graph_params, runs_params

def get_wfg_512_256_iters5_crc11():
    graph_params = {'label': 'WFG (512,256) 5 iters crc order 11', 'color': 'red', 'marker': 'o'}
    runs_params = {'code_len': 512, 'info_len': 256, 'run_name': 'wfg_512_256_iters5_crc11', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_512_256_iters5_crc11_sum_decs_2():
    graph_params = {'label': 'ensemble (512,256) 5 iters crc order 11 crc_sum%4 decs:2', 'color': 'blue', 'marker': 'x'}
    runs_params = {'code_len': 512, 'info_len': 256, 'run_name': 'ensemble_512_256_iters5_crc11_sum_decs_2', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_512_256_iters5_crc11_sum_decs_4():
    graph_params = {'label': 'ensemble (512,256) 5 iters crc order 11 crc_sum%4 decs:4', 'color': 'green', 'marker': 'x'}
    runs_params = {'code_len': 512, 'info_len': 256, 'run_name': 'ensemble_512_256_iters5_crc11_sum_decs_4', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_512_256_iters5_crc11_sum_decs_6():
    graph_params = {'label': 'ensemble (512,256) 5 iters crc order 11 crc_sum%4 decs:6', 'color': 'magenta', 'marker': 'x'}
    runs_params = {'code_len': 512, 'info_len': 256, 'run_name': 'ensemble_512_256_iters5_crc11_sum_decs_6', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_512_256_iters5_crc11_sum_decs_2_best():
    graph_params = {'label': 'ensemble (512,256) 5 iters crc order 11 crc_sum%4 decs:2 best_dec', 'color': 'blue', 'marker': '*'}
    runs_params = {'code_len': 512, 'info_len': 256, 'run_name': 'ensemble_512_256_iters5_crc11_sum_decs_2', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_512_256_iters5_crc11_sum_decs_4_best():
    graph_params = {'label': 'ensemble (512,256) 5 iters crc order 11 crc_sum%4 decs:4 best_dec', 'color': 'green', 'marker': '*'}
    runs_params = {'code_len': 512, 'info_len': 256, 'run_name': 'ensemble_512_256_iters5_crc11_sum_decs_4', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_512_256_iters5_crc11_sum_decs_6_best():
    graph_params = {'label': 'ensemble (512,256) 5 iters crc order 11 crc_sum%4 decs:6 best_dec', 'color': 'magenta', 'marker': '*'}
    runs_params = {'code_len': 512, 'info_len': 256, 'run_name': 'ensemble_512_256_iters5_crc11_sum_decs_6', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_512_256_iters6_crc11_uniform():
    graph_params = {'label': 'ensemble (512,256) 6 iters crc order 11 uniform', 'color': 'green', 'marker': 'x'}
    runs_params = {'code_len': 512, 'info_len': 256, 'run_name': 'ensemble_512_256_iters6_crc11_uniform', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_512_256_iters6_crc11_sum_mod():
    graph_params = {'label': 'ensemble (512,256) 6 iters crc order 11 sum crc mod 4', 'color': 'blue', 'marker': 'x'}
    runs_params = {'code_len': 512, 'info_len': 256, 'run_name': 'ensemble_512_256_iters6_crc11_sum', 'load_weights': True}
    return graph_params, runs_params

''' 1024 512'''

def get_polar_1024_512():
    graph_params = {'label': 'BP (1024,512) iters 5 crc11', 'color': 'black', 'marker': 'o'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'iteration_num': 5, 'load_weights': False}
    return graph_params, runs_params

def get_wfg_1024_512_iters5_crc11():
    graph_params = {'label': 'WFG (1024,512) 5 iters crc order 11', 'color': 'red', 'marker': 'o'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'run_name': 'wfg_1024_512_iters5_crc11', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_1024_512_iters5_crc11_sum_decs_4():
    graph_params = {'label': 'ensemble (1024,512) 5 iters crc order 11 crc_sum%4 decs:4', 'color': 'blue', 'marker': 'x'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'run_name': 'ensemble_1024_512_iters5_crc11_sum_decs_4', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_1024_512_iters5_crc11_sum_decs_4_best():
    graph_params = {'label': 'ensemble (1024,512) 5 iters crc order 11 crc_sum%4 decs:4 best_dec', 'color': 'green', 'marker': '*'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'run_name': 'ensemble_1024_512_iters5_crc11_sum_decs_4', 'load_weights': True}
    return graph_params, runs_params

def get_wfg_1024_512_iters6_crc11():
    graph_params = {'label': 'WFG (1024,512) 6 iters crc order 11', 'color': 'red', 'marker': 'o'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'run_name': 'wfg_1024_512_iters6_crc11', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_1024_512_iters6_crc11_sum_decs_4():
    graph_params = {'label': 'ensemble (1024,512) 6 iters crc order 11 crc_sum%4 decs:4', 'color': 'blue', 'marker': 'x'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'run_name': 'ensemble_1024_512_iters6_crc11_sum_decs_4', 'load_weights': True}
    return graph_params, runs_params

def get_ensemble_1024_512_iters6_crc11_sum_decs_4_best():
    graph_params = {'label': 'ensemble (1024,512) 6 iters crc order 11 crc_sum%4 decs:4 best_dec', 'color': 'green', 'marker': '*'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'run_name': 'ensemble_1024_512_iters6_crc11_sum_decs_4', 'load_weights': True}
    return graph_params, runs_params