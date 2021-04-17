from python_code.polar_codes.initialization import initialize_factor_graph, initialize_polar_code
from python_code.crc_codes.crc import create_crc_matrices
from python_code.tanner.load import load_code_parameters
from python_code.polar_codes.stop_condition import *
from python_code.tanner.bp_nn import *
import numpy as np
import torch

MULTILOSS_MASK = True
OUTPUT_MASK = False
EVEN_MASK = True
ODD_MASK = True
EARLY_TERMINATION_TRAIN = False
EARLY_TERMINATION_EVAL = True
NN_MODEL = 'RNN'


class TannerDecoder(torch.nn.Module):
    def __init__(self, code_len, info_len, design_SNR, crc, iteration_num, clipping_val, device):
        super(TannerDecoder, self).__init__()

        self.code_gm = initialize_factor_graph(code_len, device=device)
        self.info_ind, self.crc_ind = initialize_polar_code(code_len, info_len, design_SNR, crc, device)
        self.code_pcm = self.code_gm[:, ~self.info_ind].T
        self.code_pcm = self.code_pcm.cpu().numpy().astype(np.float32)
        self.crc_pcm, self.crc_gm = create_crc_matrices(info_len, crc, eye='before', device=device)
        self.clipping_val = clipping_val
        self.device = device

        # Neural Network config
        self.input_output_layer_size = code_len
        self.iteration_num = iteration_num
        self.neurons = int(np.sum(self.code_pcm))

        # define layers
        if NN_MODEL == 'FC':
            self.odd_layer = torch.nn.ModuleList()
            self.multiloss_output_layer = torch.nn.ModuleList()
            self.multiloss_output_layer.append(OutputLayer(neurons=self.neurons,
                                                           input_output_layer_size=self.input_output_layer_size,
                                                           code_pcm=self.code_pcm))
            self.even_layer = torch.nn.ModuleList()
            self.input_layer = InputLayer(input_output_layer_size=self.input_output_layer_size, neurons=self.neurons,
                                          code_pcm=self.code_pcm, clip_tanh=self.clipping_val,
                                          bits_num=code_len)
            for i in range(self.iteration_num - 1):
                self.odd_layer.append(OddLayer(clip_tanh=self.clipping_val,
                                               input_output_layer_size=self.input_output_layer_size,
                                               neurons=self.neurons,
                                               code_pcm=self.code_pcm))
                self.even_layer.append(EvenLayer(self.clipping_val, self.neurons, self.code_pcm))
                self.multiloss_output_layer.append(OutputLayer(neurons=self.neurons,
                                                               input_output_layer_size=self.input_output_layer_size,
                                                               code_pcm=self.code_pcm))
            self.output_layer = OutputLayer(neurons=self.neurons,
                                            input_output_layer_size=self.input_output_layer_size,
                                            code_pcm=self.code_pcm)

        elif NN_MODEL == 'RNN':
            self.input_layer = InputLayer(input_output_layer_size=self.input_output_layer_size, neurons=self.neurons,
                                          code_pcm=self.code_pcm, clip_tanh=self.clipping_val,
                                          bits_num=code_len)
            self.even_layer = EvenLayer(self.clipping_val, self.neurons, self.code_pcm)
            self.odd_layer = OddLayer(clip_tanh=self.clipping_val,
                                      input_output_layer_size=self.input_output_layer_size,
                                      neurons=self.neurons,
                                      code_pcm=self.code_pcm)
            self.multiloss_output_layer = OutputLayer(neurons=self.neurons,
                                                      input_output_layer_size=self.input_output_layer_size,
                                                      code_pcm=self.code_pcm)
            self.output_layer = OutputLayer(neurons=self.neurons,
                                            input_output_layer_size=self.input_output_layer_size,
                                            code_pcm=self.code_pcm)
        else:
            raise Exception('nn_model {} is not implemented'.format(NN_MODEL))

    def forward(self, x):
        """
        compute forward pass in the network
        :param x: [batch_size,N]
        :return: decoded word [batch_size,N]
        """
        # initialize parameters
        output_list = [0] * (self.iteration_num + 1)
        not_satisfied_list = [0] * (self.iteration_num - 1)
        not_satisfied = torch.arange(x.size(0), dtype=torch.long, device=self.device)
        output_list[-1] = torch.zeros_like(x)

        # equation 1 and 2 from "Learning To Decode ..", i==1,2 (iteration 1)
        even_output = self.input_layer.forward(x)
        if NN_MODEL == 'FC':
            output_list[0] = torch.index_select(x, 0, not_satisfied) + self.multiloss_output_layer[0].forward(
                even_output[not_satisfied], mask_only=MULTILOSS_MASK)
        elif NN_MODEL == 'RNN':
            output_list[0] = torch.index_select(x, 0, not_satisfied) + self.multiloss_output_layer.forward(
                even_output[not_satisfied], mask_only=MULTILOSS_MASK)

        # now start iterating through all hidden layers i>2 (iteration 2 - Imax)
        for i in range(0, self.iteration_num - 1):
            # odd - variables to check
            if NN_MODEL == 'FC':
                odd_output_not_satisfied = self.odd_layer[i].forward(torch.index_select(even_output, 0, not_satisfied),
                                                                     torch.index_select(x, 0, not_satisfied),
                                                                     llr_mask_only=ODD_MASK)
                # even - check to variables
                even_output[not_satisfied] = self.even_layer[i].forward(odd_output_not_satisfied,
                                                                        mask_only=EVEN_MASK)
                # output layer
                output_not_satisfied = torch.index_select(x, 0, not_satisfied) + self.multiloss_output_layer[
                    i + 1].forward(
                    even_output[not_satisfied], mask_only=MULTILOSS_MASK)
                output_list[i + 1] = output_not_satisfied.clone()
                not_satisfied_list[i] = not_satisfied.clone()

            elif NN_MODEL == 'RNN':
                odd_output_not_satisfied = self.odd_layer.forward(torch.index_select(even_output, 0, not_satisfied),
                                                                  torch.index_select(x, 0, not_satisfied),
                                                                  llr_mask_only=ODD_MASK)
                # even - check to variables
                even_output[not_satisfied] = self.even_layer.forward(odd_output_not_satisfied,
                                                                     mask_only=EVEN_MASK)
                # output layer
                output_not_satisfied = torch.index_select(x, 0, not_satisfied) + self.multiloss_output_layer.forward(
                    even_output[not_satisfied], mask_only=MULTILOSS_MASK)
                output_list[i + 1] = output_not_satisfied.clone()
                not_satisfied_list[i] = not_satisfied.clone()

            if EARLY_TERMINATION_TRAIN and output_not_satisfied.requires_grad:
                not_satisfied = hard_decision_condition(not_satisfied, output_not_satisfied)
            if EARLY_TERMINATION_EVAL and not output_not_satisfied.requires_grad:
                output_list[-1][not_satisfied] = output_not_satisfied.clone()
                not_satisfied = syndrome_condition(not_satisfied, output_not_satisfied, self.code_pcm)
            if not_satisfied.size(0) == 0:
                break
        output_list[-1][not_satisfied] = x[not_satisfied] + self.output_layer.forward(even_output[not_satisfied],
                                                                                      mask_only=OUTPUT_MASK)
        return output_list, not_satisfied_list
