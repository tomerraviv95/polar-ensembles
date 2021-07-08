from python_code.codes.polar_codes.polar_nn import IterateRightLayer, IterateLeftLayer
from python_code.codes.polar_codes.stop_condition import generator_criterion
from python_code.utils.python_utils import llr_to_bits
from python_code.decoders.decoder import Decoder
import numpy as np
import torch

EARLY_TERMINATION_EVAL = True


class FGDecoder(Decoder):
    """
    The decoder class. Calls the graph building process and implements the decoding algorithm.
    See that after every decoding iteration (L->R>L) we usually stop decoding if some stopping criterion holds.
    """

    def __init__(self, code_len, info_len, design_snr, clipping_val, iteration_num, device):
        super().__init__(code_len, info_len, design_snr, clipping_val, iteration_num, device)

        self.num_stages = int(np.log2(code_len))
        self._build_model()

    def _build_model(self):
        # define layers
        self.iterate_right_layer = IterateRightLayer(self.code_len, self.clipping_val, self.iteration_num)
        self.iterate_left_layer = IterateLeftLayer(self.code_len, self.clipping_val, self.iteration_num)

    def _initialize_graph(self, x):
        """
        initializes the network - defines L_init and matrices for both R and L
        """
        right = torch.zeros([x.size(0), self.num_stages + 1, self.code_len], device=self.device)
        right[:, 0] = (1 - self.total_info_bits.float()) * self.clipping_val
        left = torch.zeros_like(right)
        left[:, -1] = x
        return right, left

    def forward(self, rx: torch.Tensor):
        """
        compute forward pass in the network
        :param rx: [batch_size,N]

        A note here is the not_satisfied vector.
        See that it holds all positions for the unsatisfied words and we save it every iterations.
        We return the list of decoded words and unsatisfied words for every iteration
        """
        # initialize parameters
        u_list = [0] * (self.iteration_num + 1)
        u_list[-1] = torch.zeros((rx.size(0), self.info_len), device=rx.device)

        not_satisfied_list = [0] * self.iteration_num
        not_satisfied = torch.arange(rx.size(0), dtype=torch.long, device=self.device)

        # initialize graph
        right, left = self._initialize_graph(rx)

        for i in range(self.iteration_num):
            # iterate right
            right[not_satisfied] = self.iterate_right_layer(right[not_satisfied], left[not_satisfied], i)

            # iterate left
            left[not_satisfied] = self.iterate_left_layer(right[not_satisfied], left[not_satisfied], i)

            u = left[not_satisfied, 0] + right[not_satisfied, 0]
            x = left[not_satisfied, self.num_stages] + right[not_satisfied, self.num_stages]

            u_list[i] = u[:, self.info_ind]

            not_satisfied_list[i] = not_satisfied.clone()
            u_list[-1][not_satisfied] = u_list[i].clone()
            if EARLY_TERMINATION_EVAL:
                not_satisfied = generator_criterion(llr_to_bits(x), llr_to_bits(u), self.code_gm,
                                                    not_satisfied)
            if not_satisfied.size(0) == 0:
                break
        return u_list, not_satisfied_list
