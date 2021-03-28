import numpy as np


def BPSKmodulation(x: np.array):
    """
    BPSK modulation 0->1, 1->-1
    Input: x - Encoded matrix with Batch x N
    Output: tx - Modulated matrix
    """
    tx = 1 - 2 * x
    return tx


def AWGN(tx: np.array, SNR: float, R: float, random: np.random.RandomState, use_llr: bool = True) -> np.array:
    """ 
        Input: tx - Transmitted codeword, SNR - dB, R - Code rate, use_llr - Return llr
        Output: rx - Codeword with AWGN noise
    """
    [row, col] = tx.shape

    sigma = np.sqrt(0.5 * ((10 ** ((SNR + 10 * np.log10(R)) / 10)) ** (-1)))

    rx = tx + sigma * random.normal(0.0, 1.0, (row, col))

    if use_llr:
        return 2 * rx / (sigma ** 2)
    else:
        return rx
