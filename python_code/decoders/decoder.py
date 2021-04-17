from python_code.codes.polar_codes import initialize_factor_graph, initialize_polar_code
from python_code.codes.crc_codes.crc import create_crc_matrices
from typing import List
import numpy as np
import torch


class Decoder(torch.nn.Module):
    """Class for generic decoder"""

    def __init__(self, code_len, info_len, design_snr, clipping_val, iteration_num, crc, device):
        super(Decoder, self).__init__()
        self.code_len: int = code_len
        self.info_len: int = info_len
        self.design_snr: float = design_snr
        self.clipping_val: float = clipping_val
        self.iteration_num: int = iteration_num
        self.crc: List[int] = crc
        self.device: torch.device = device
        self.code_gm = initialize_factor_graph(self.code_len, self.device)
        self.info_ind, self.crc_ind = initialize_polar_code(self.code_len, self.info_len, self.design_snr, self.crc,
                                                            self.device)
        self.code_pcm = self.code_gm[:, ~self.info_ind].T
        self.code_pcm = self.code_pcm.cpu().numpy().astype(np.float32)
        if len(self.crc):
            self.total_info_bits = (self.info_ind.float() + self.crc_ind.float()).byte()
        else:
            self.total_info_bits = self.info_ind
        self.poly = torch.tensor(self.crc, device=self.device).byte()
        self.crc_pcm, self.crc_gm = create_crc_matrices(self.info_len, self.crc, eye='before', device=self.device)
