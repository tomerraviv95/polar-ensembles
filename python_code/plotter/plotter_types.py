def get_polar_64_32():
    graph_params = {'label': 'FG (64,32)', 'color': 'black', 'marker': 'o'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'polar_fg_64_32', 'load_weights': False}
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
    graph_params = {'label': 'ensemble (64,32) 6 iters crc order 11 sum crc mod 4', 'color': 'green', 'marker': 'o'}
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

def get_polar_128_64():
    graph_params = {'label': 'FG (128,64)', 'color': 'blue', 'marker': 'o'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'polar_fg_128_64', 'load_weights': False}
    return graph_params, runs_params


def get_weighted_polar_128_64():
    graph_params = {'label': 'WFG (128,64)', 'color': 'blue', 'marker': 'x'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'polar_fg_128_64', 'load_weights': True}
    return graph_params, runs_params


def get_polar_256_128():
    graph_params = {'label': 'FG (256,128)', 'color': 'black', 'marker': 'o'}
    runs_params = {'code_len': 256, 'info_len': 128, 'run_name': 'polar_fg_256_128', 'load_weights': False}
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
    graph_params = {'label': 'ensemble (256,128) 6 iters crc order 11, best dec', 'color': 'green', 'marker': 'x'}
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