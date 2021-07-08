import torch
import numpy as np


def initialize_factor_graph(code_len: int, device: torch.device) -> torch.Tensor:
    """
    Return the G matrix of the code, this is the nth kroncker product of the F matrix
    """
    factor_graph = np.array([[1, 0], [1, 1]])
    n_factor = factor_graph.copy()
    for _ in range(int(np.log2(code_len)) - 1):
        n_factor = np.kron(n_factor, factor_graph)
    return torch.tensor(n_factor).to(device=device).float()


def calculate_z_parameters_one_recursion(z_params):
    """
    Construction based on the Bhattacharyya bounds, see Arikan for further details
    """
    z_next = np.empty(2 * z_params.size, dtype=z_params.dtype)
    z_sq = z_params ** 2
    z_low = 2 * z_params - z_sq
    z_next[0::2] = z_low
    z_next[1::2] = z_sq
    return z_next


def initialize_polar_code(code_len: int, info_len: int, design_SNR: float, device: torch.device):
    """
    Output:
    info-bit indices [info_len]
    subset of {0, 1, . . . , code_len - 1}
    with |A==1| = info_len logical [1 X N]
    """
    n = np.ceil(np.log2(code_len)).astype(int)
    z0 = np.array(np.exp(-10 ** (design_SNR / 10)))
    for j in range(n):
        z0 = calculate_z_parameters_one_recursion(z0)

    # Find greatest (N-K) elements to  frozen bits
    sort_idx = np.argsort(-z0)
    sort_idx = sort_idx[sort_idx < code_len]
    info_index = sort_idx[code_len - info_len:]
    info_mat = np.full(code_len, False, dtype=bool)
    info_mat[info_index] = True

    return torch.tensor(info_mat, device=device)


def initialize_connections(code_len: int) -> np.array:
    """
    creates the connections matrix of the factor graph
    of size [code_lem,log2(code_len)]
    """
    stages = int(np.log2(code_len))
    j0_mat = np.zeros((stages, int(code_len / 2)))
    j1_mat = np.zeros((stages, int(code_len / 2)))

    j0_mat[0, :] = np.arange(int(code_len / 2))
    j1_mat[0, :] = np.arange(int(code_len / 2)) + int(code_len / 2)

    for i in range(stages - 1):
        j0_mat[i + 1, :] = np.reshape(
            np.stack((j0_mat[i, : int(code_len / 4)], j1_mat[i, : int(code_len / 4)]), axis=1),
            (1, int(code_len / 2)))
        j1_mat[i + 1, :] = np.reshape(np.stack((j0_mat[i, int(code_len / 4):], j1_mat[i, int(code_len / 4):]), axis=1),
                                      (1, int(code_len / 2)))
    connections = np.ones((stages, code_len))
    for i in range(stages):
        connections[i, j1_mat[i, :].astype(int)] = 0

    connections = np.flipud(connections)

    return connections
