from typing import Dict
import pickle as pkl
import torch


def save_pkl(pkls_path: str, array: Dict):
    output = open(pkls_path, 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str):
    output = open(pkls_path, 'rb')
    return pkl.load(output)


def llr_to_bits(x):
    """
    num>0 -> 0
    num<0 -> 1
    """
    return torch.round(torch.sigmoid(-x))
