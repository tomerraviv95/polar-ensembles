import pickle as pkl
from typing import Dict

import numpy as np


def save_pkl(pkls_path: str, array: Dict):
    output = open(pkls_path, 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str):
    output = open(pkls_path, 'rb')
    return pkl.load(output)
