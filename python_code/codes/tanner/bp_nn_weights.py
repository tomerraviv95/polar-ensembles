import numpy as np
import torch


def initialize_w_init(input_output_layer_size, neurons, code_pcm):
    """
    intialize input layer weights
    :return: weights of size [N,neurons]
    """
    w_input = np.zeros((input_output_layer_size, neurons), dtype=np.float32)
    k = 0
    for i in range(code_pcm.shape[0]):
        for j in range(code_pcm.shape[1]):
            if code_pcm[i, j] == 1:
                vec = code_pcm[i, :].copy()
                vec[j] = 0
                w_input[:, k] = vec
                k += 1
    return torch.tensor(w_input)


def initialize_w_v2c(neurons, code_pcm):
    k = 0
    w_odd2even = np.zeros((neurons, neurons), dtype=np.float32)
    for j in range(0, code_pcm.shape[1]):  # run over the columns
        for i in range(0, code_pcm.shape[0]):  # break after the first one
            if code_pcm[i, j] == 1:
                num_of_conn = np.sum(code_pcm[:, j])  # get the number of connection of the variable node
                idx = np.argwhere(code_pcm[:, j] == 1)  # get the indexes
                for l in range(0, int(num_of_conn)):  # adding num_of_conn columns to W
                    vec_tmp = np.zeros(neurons, dtype=np.float32)
                    for r in range(0, code_pcm.shape[0], 1):  # adding one to the right place
                        if code_pcm[r, j] == 1 and idx[l][0] != r:
                            idx_vec = np.cumsum(code_pcm[r, 0:j + 1])[-1] - 1
                            vec_tmp[int(idx_vec + np.sum(code_pcm[:r, :]))] = 1.0
                    w_odd2even[:, k] = vec_tmp.transpose()
                    k += 1
                break

    return torch.tensor(w_odd2even, requires_grad=True), torch.tensor(w_odd2even, requires_grad=False)


def initialize_w_c2v(neurons, code_pcm):
    k = 0
    w_even2odd = np.zeros((neurons, neurons), dtype=np.float32)
    for j in range(0, code_pcm.shape[1]):
        for i in range(0, code_pcm.shape[0]):
            if code_pcm[i, j] == 1:
                idx_row = np.cumsum(code_pcm[i, 0:j + 1])[-1] - 1
                untill_d_c = np.sum(code_pcm[:i, :])
                this_d_c = np.sum(code_pcm[:(i + 1), :])
                w_even2odd[k, int(untill_d_c):int(this_d_c)] = 1.0
                w_even2odd[k, int(untill_d_c + idx_row)] = 0.0
                k += 1

    return torch.tensor(w_even2odd, requires_grad=True), torch.tensor(w_even2odd, requires_grad=False)


def init_w_output(neurons, input_output_layer_size, code_pcm):
    k = 0
    w_output = np.zeros((neurons, input_output_layer_size), dtype=np.float32)
    for j in range(0, code_pcm.shape[1]):
        for i in range(0, code_pcm.shape[0]):
            if code_pcm[i, j] == 1:
                idx_row = np.cumsum(code_pcm[i, 0:j + 1])[-1] - 1
                untill_d_c = np.sum(code_pcm[:i, :])
                w_output[int(untill_d_c + idx_row), k] = 1.0
        k += 1

    return torch.tensor(w_output, requires_grad=True), torch.tensor(w_output, requires_grad=False)


def init_w_skipconn2even(input_output_layer_size, neurons, code_pcm):
    w_skipconn2even = np.zeros((input_output_layer_size, neurons), dtype=np.float32)
    k = 0
    for j in range(0, code_pcm.shape[1]):
        for i in range(0, code_pcm.shape[0]):
            if code_pcm[i, j] == 1:
                w_skipconn2even[j, k] = 1.0
                k += 1
    return torch.tensor(w_skipconn2even, requires_grad=True), torch.tensor(w_skipconn2even, requires_grad=False)
