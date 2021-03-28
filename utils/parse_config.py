import argparse
import yaml
import os


def parse_config_file() -> dict:
    """
    Read config from cmd, current folder or parent folder
    Then load it into a dict, which is returned
    :return:
    """
    parser = argparse.ArgumentParser()
    try:
        parser.add_argument('--configuration', type=str, help='yaml configuration file.', required=True)
        args = parser.parse_args()
        configuration = args.configuration
    except SystemExit:
        # check current folder for config files
        current_files = [f for f in os.listdir('.')]
        if 'config.yaml' in current_files:
            configuration = 'config.yaml'
        else:
            # check parent folder for config files
            parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            parent_files = [f for f in os.listdir(parent_path)]
            if 'config.yaml' in parent_files:
                configuration = '../config.yaml'
    try:
        with open(configuration) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    except Exception:
        raise Exception("Did not specify config file")

    return config
