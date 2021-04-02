def get_polar_64_32():
    graph_params = {'label': 'FG (64,32)', 'color': 'red', 'marker': 'o'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'polar_fg_64_32', 'load_weights': False}
    return graph_params, runs_params


def get_weighted_polar_64_32():
    graph_params = {'label': 'WFG (64,32)', 'color': 'red', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'polar_fg_64_32', 'load_weights': True}
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
    runs_params = {'code_len': 256, 'info_len': 128, 'run_name': 'polar_fg_256_128', 'load_weights': True}
    return graph_params, runs_params


def get_polar_1024_512():
    graph_params = {'label': 'FG (1024,512)', 'color': 'green', 'marker': 'o'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'run_name': 'polar_fg_1024_512', 'load_weights': False}
    return graph_params, runs_params


def get_weighted_polar_1024_512():
    graph_params = {'label': 'WFG (1024,512)', 'color': 'green', 'marker': 'x'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'run_name': 'polar_fg_1024_512', 'load_weights': True}
    return graph_params, runs_params
