def get_polar_64_32():
    graph_params = {'label': 'Polar Code (64,32)', 'color': 'red', 'marker': 'o'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'polar_fg_64_32', 'load_weights': False}
    return graph_params, runs_params


def get_weighted_polar_64_32():
    graph_params = {'label': 'Weighted Polar Code (64,32)', 'color': 'red', 'marker': 'x'}
    runs_params = {'code_len': 64, 'info_len': 32, 'run_name': 'polar_fg_64_32', 'load_weights': True}
    return graph_params, runs_params


def get_polar_128_64():
    graph_params = {'label': 'Polar Code (128,64)', 'color': 'blue', 'marker': 'o'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'polar_fg_128_64', 'load_weights': False}
    return graph_params, runs_params


def get_weighted_polar_128_64():
    graph_params = {'label': 'Weighted Polar Code (128,64)', 'color': 'blue', 'marker': 'x'}
    runs_params = {'code_len': 128, 'info_len': 64, 'run_name': 'polar_fg_128_64', 'load_weights': True}
    return graph_params, runs_params


def get_polar_1024_512():
    graph_params = {'label': 'Polar Code (1024,512)', 'color': 'green', 'marker': 'o'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'run_name': 'polar_fg_1024_512', 'load_weights': False}
    return graph_params, runs_params


def get_weighted_polar_1024_512():
    graph_params = {'label': 'Weighted Polar Code (1024,512)', 'color': 'green', 'marker': 'x'}
    runs_params = {'code_len': 1024, 'info_len': 512, 'run_name': 'polar_fg_1024_512', 'load_weights': True}
    return graph_params, runs_params

def get_polar_2048_1024():
    graph_params = {'label': 'Polar Code (2048,1024)', 'color': 'black', 'marker': 'o'}
    runs_params = {'code_len': 2048, 'info_len': 1024, 'run_name': 'polar_fg_2048_1024', 'load_weights': False}
    return graph_params, runs_params


def get_weighted_polar_2048_1024():
    graph_params = {'label': 'Weighted Polar Code (2048,1024)', 'color': 'black', 'marker': 'x'}
    runs_params = {'code_len': 2048, 'info_len': 1024, 'run_name': 'polar_fg_2048_1024', 'load_weights': True}
    return graph_params, runs_params
