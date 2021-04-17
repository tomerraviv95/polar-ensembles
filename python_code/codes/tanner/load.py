from python_code.codes.polar_codes import initialize_factor_graph, initialize_polar_code
import torch


def load_code_parameters(bits_num, parity_bits_num, design_SNR):
    code_gm = initialize_factor_graph(bits_num, device=torch.device("cuda"))
    CRC = []
    A_info, A_CRC = initialize_polar_code(bits_num, parity_bits_num, design_SNR, CRC, device=torch.device("cuda"))
    code_pcm = code_gm[~A_info]
    t = 0
    return code_pcm, code_gm, t
