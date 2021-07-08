from python_code.codes.polar_codes.initialization import initialize_factor_graph, initialize_polar_code
import numpy as np
import torch


class Decoder(torch.nn.Module):
    """Class for generic decoder"""

    def __init__(self, code_len, info_len, design_snr, clipping_val, iteration_num, device):
        super(Decoder, self).__init__()
        self.code_len: int = code_len
        self.info_len: int = info_len
        self.design_snr: float = design_snr
        self.clipping_val: float = clipping_val
        self.iteration_num: int = iteration_num
        self.device: torch.device = device
        self.code_gm = initialize_factor_graph(self.code_len, self.device)
        self.info_ind = initialize_polar_code(self.code_len, self.info_len, self.design_snr, self.device)
        self.code_pcm = self.code_gm[:, ~self.info_ind].T
        self.code_pcm = self.code_pcm.cpu().numpy().astype(np.float32)
        self.total_info_bits = self.info_ind
