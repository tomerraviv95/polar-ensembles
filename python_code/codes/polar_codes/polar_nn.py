from globals import DEVICE
import torch
import torch.nn as nn
import numpy as np

from python_code.codes.polar_codes.initialization import initialize_connections


def min_sum(x, y):
    """
    The approximation used in the message passing algorithm
    """
    return torch.sign(x) * torch.sign(y) * torch.min(torch.abs(x), torch.abs(y))


def get_masks_dicts(code_len, connections, pi=None):
    """
    messages are propagated from each stage to the next in a vectorized manner
    using the masks below. see the pi controls the order of the connections mat
    that is - changing pi is necessary for permutations of the layers
    """
    l_stages = int(np.log2(code_len))
    if not pi:
        pi = list(range(l_stages))
    maskt = torch.BoolTensor(connections[pi, :].T.astype(float)).to(device='cpu')
    mask_resized_dict = {}
    negative_mask_resized_dict = {}
    for i in range(l_stages):
        mask = maskt[:, i]
        mask_resized_dict[i] = torch.masked_select(torch.arange(len(mask)), mask.bool()).long().view(code_len // 2)
        negative_mask_resized_dict[i] = torch.masked_select(torch.arange(len(mask)), ~mask.bool()).long().view(
            code_len // 2)
    return mask_resized_dict, negative_mask_resized_dict


class IterateRightLayer(torch.nn.Module):
    """
    Right message passing layer
    """

    def __init__(self, code_len, clipping_val, num_hidden_layers):
        super().__init__()
        self.clipping_val = clipping_val
        # initialize weights
        connections = initialize_connections(code_len)
        self.mask_dict, self.negative_mask_dict = get_masks_dicts(code_len, connections)
        self.num_stages = int(np.log2(code_len))
        self.right_weights = nn.Parameter(torch.ones((num_hidden_layers, self.num_stages, 2), device=DEVICE))

    def forward(self, right, left, iter):
        for i in range(self.num_stages):
            left_prev0 = left[:, i + 1, self.negative_mask_dict[i]]
            left_prev1 = left[:, i + 1, self.mask_dict[i]]
            right_prev0 = right[:, i, self.negative_mask_dict[i]]
            right_prev1 = right[:, i, self.mask_dict[i]]
            right[:, i + 1, self.mask_dict[i]] = self.right_weights[iter, i, 0] * min_sum(right_prev1,
                                                                                          left_prev0 + right_prev0)
            right[:, i + 1, self.negative_mask_dict[i]] = self.right_weights[iter, i, 1] * min_sum(right_prev1,
                                                                                                   left_prev1) + right_prev0
        right = torch.clamp(right, -self.clipping_val, self.clipping_val)
        return right


class IterateLeftLayer(torch.nn.Module):
    """
    Left message passing layer
    """

    def __init__(self, code_len, clipping_val, num_hidden_layers):
        super().__init__()
        self.clipping_val = clipping_val
        # initialize weights
        connections = initialize_connections(code_len)
        self.mask_dict, self.negative_mask_dict = get_masks_dicts(code_len, connections)
        self.num_stages = int(np.log2(code_len))
        self.left_weights = nn.Parameter(torch.ones((num_hidden_layers, self.num_stages, 2), device=DEVICE))

    def forward(self, right, left, iter):
        for i in reversed(range(self.num_stages)):
            left_prev0 = left[:, i + 1, self.negative_mask_dict[i]]
            left_prev1 = left[:, i + 1, self.mask_dict[i]]
            right_prev0 = right[:, i, self.negative_mask_dict[i]]
            right_prev1 = right[:, i, self.mask_dict[i]]

            left[:, i, self.mask_dict[i]] = self.left_weights[iter, i, 0] * min_sum(left_prev1,
                                                                                    left_prev0 + right_prev0)
            left[:, i, self.negative_mask_dict[i]] = self.left_weights[iter, i, 1] * min_sum(left_prev1,
                                                                                             right_prev1) + left_prev0

        left = torch.clamp(left, -self.clipping_val, self.clipping_val)
        return left
