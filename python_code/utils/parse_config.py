from dir_definitions import CONFIG_PATH
import yaml


def parse_config_file() -> dict:
    """
    Read config from cmd, current folder or parent folder
    Then load it into a dict, which is returned
    :return:
    """
    with open(CONFIG_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
