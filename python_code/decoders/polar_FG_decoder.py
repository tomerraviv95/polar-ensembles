import torch
import numpy as np
from python_code.basic.polar_nn import IterateLeftLayer, IterateRightLayer
from python_code.polar_codes.initialization import initialize_factor_graph, initialize_polar_code
from python_code.polar_codes.stop_condition import crc_criterion, generator_criterion
from python_code.crc_codes.crc import create_crc_matrices


def llr_to_bits(x):
    """
    num>0 -> 0
    num<0 -> 1
    """
    return torch.round(torch.sigmoid(-x))


class PolarFGDecoder(torch.nn.Module):
    """
    The decoder class. Calls the graph building process and implements the decoding algorithm.
    See that after every decoding iteration (L->R>L) we usually stop decoding if some stopping criterion holds.
    """
    def __init__(self, code_len, info_len, design_SNR, crc, iteration_num, clipping_val=15,
                 filter_in_iterations_eval=False, device='cpu'):
        super().__init__()

        self.device = device
        # code params
        self.code_len = code_len
        self.num_stages = int(np.log2(code_len))
        self.info_len = info_len
        self.design_snr = design_SNR
        self.clipping_val = clipping_val
        self.factor_graph = initialize_factor_graph(code_len, device)
        self.info_ind, self.crc_ind = initialize_polar_code(code_len, info_len, design_SNR, crc, device=device)
        if len(crc):
            self.total_info_bits = (self.info_ind.float() + self.crc_ind.float()).byte()
        else:
            self.total_info_bits = self.info_ind
        self.poly = torch.tensor(crc, device=device).byte()
        self.crc_pcm, self.crc_gm = create_crc_matrices(info_len, crc, eye='before', device=device)

        # model parameters
        self.num_hidden_layers = iteration_num
        self.filter_in_iterations_eval = filter_in_iterations_eval
        self._build_model()

    def _build_model(self):
        # define layers
        self.iterate_right_layer = IterateRightLayer(self.code_len, self.clipping_val, device=self.device)
        self.iterate_left_layer = IterateLeftLayer(self.code_len, self.clipping_val, device=self.device)

    def _initialize_graph(self, x, total_info_bits):
        """
        initializes the network - defines L_init and matrices for both R and L
        """
        right = torch.zeros([x.size(0), self.num_stages + 1, self.code_len], device=self.device)
        right[:, 0] = (1 - total_info_bits.float()) * self.clipping_val

        left = torch.zeros_like(right)
        left[:, -1] = x
        return right, left

    def forward(self, rx:torch.Tensor):
        """
        compute forward pass in the network
        :param rx: [batch_size,N]

        A note here is the not_satisfied vector.
        See that it holds all positions for the unsatisfied words and we save it every iterations.
        We return the list of decoded words and unsatisfied words for every iteration
        """
        # initialize parameters
        u_list = [0] * (self.num_hidden_layers + 1)
        u_list[-1] = torch.zeros((rx.size(0), self.info_len), device=rx.device)

        not_satisfied_list = [0] * self.num_hidden_layers
        not_satisfied = torch.arange(rx.size(0), dtype=torch.long, device=self.device)

        # permute info bits indices
        total_info_bits = self.total_info_bits

        # initialize graph
        right, left = self._initialize_graph(rx, total_info_bits)

        for i in range(self.num_hidden_layers):
            # iterate right
            right[not_satisfied] = self.iterate_right_layer(right[not_satisfied], left[not_satisfied])

            # iterate left
            left[not_satisfied] = self.iterate_left_layer(right[not_satisfied], left[not_satisfied])

            u = left[not_satisfied, 0] + right[not_satisfied, 0]
            x = left[not_satisfied, self.num_stages] + right[not_satisfied, self.num_stages]

            u_list[i] = u[:, self.info_ind]

            not_satisfied_list[i] = not_satisfied.clone()
            u_list[-1][not_satisfied] = u_list[i].clone()

            if self.filter_in_iterations_eval:
                if len(self.poly):
                    not_satisfied = crc_criterion(llr_to_bits(u), self.info_ind, self.crc_ind, self.poly,
                                                  self.crc_gm, not_satisfied)
                else:
                    not_satisfied = generator_criterion(llr_to_bits(x), llr_to_bits(u), self.factor_graph,
                                                        not_satisfied)
            if not_satisfied.size(0) == 0:
                break
        return u_list, not_satisfied_list
