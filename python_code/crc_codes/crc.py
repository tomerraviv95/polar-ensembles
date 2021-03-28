import torch
import numpy as np


def encode_by_generator_matrix(words, poly, crc_gm):
    """
    Returns CRC of current words, used for the G-based stopping criterion
    """
    encoded_words = torch.fmod(torch.mm(words.float(), crc_gm.t().float()), 2)
    return encoded_words[:, :len(poly)]


def create_crc_matrices(info_len: int, crc: list, eye: str, device: torch.device) -> (
torch.ByteTensor, torch.ByteTensor):
    """
    Creates the G matrix for the CRC code (which encodes the info bits)
    Allows the G matrix to have the CRC before or after the systematic part
    :param info_len: information length
    :param crc: the polynomial
    :param eye: before or after
    :param device: device
    """
    crc = np.flip(np.array(crc), axis=0)
    k = len(crc) + info_len
    gm = np.zeros((info_len, k))
    eye_mat = np.eye(info_len, info_len)
    gm[0] = np.concatenate((crc, eye_mat[0]))

    for i in range(1, info_len):
        gm[i, :k - info_len] = np.roll(gm[i - 1, :k - info_len], 1)
        gm[i, 0] = 0
        if gm[i - 1, k - info_len - 1] == 1:
            gm[i, :k - info_len] += crc
            gm = np.mod(gm, 2)
        gm[i, k - info_len:] = eye_mat[i]

    if eye == "before":
        pcm = np.concatenate((np.eye(k - info_len, k - info_len), gm[:, :k - info_len].T), axis=1)
    else:
        pcm = np.concatenate((gm[:, :k - info_len].T, np.eye(k - info_len, k - info_len)), axis=1)
    return torch.tensor(pcm, device=device).byte(), torch.tensor(gm.T, device=device).byte()
