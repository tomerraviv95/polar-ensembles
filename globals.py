from python_code.utils.config_singleton import Config
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = Config()
