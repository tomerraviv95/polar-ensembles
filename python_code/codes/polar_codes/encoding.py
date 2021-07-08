import numpy as np


def encode(target, code_gm, code_len, info_ind):
    """
    Input: u - Information matrix with Batch x info (each row is information bits), G - Generator matrix with info x N
    Output: x - Encoded matrix Batch x code_len
    """
    batch_size, _ = target.shape
    u = np.zeros((batch_size, code_len))
    u[:, info_ind] = target
    x = transform_by_mat(u, code_gm)
    return x


# Non-systematic polar encoding
def transform_by_recursion(u):
    """ 
        Input: u - same as above
        Output: x - same as above
    """
    _, N = u.shape

    for i in range(int(np.log2(N))):
        temp = -1 * np.ones(N)  # In order not to visit a bit twice
        separation = int(N / 2 ** (i + 1))

        for j in range(N):
            if temp[j] == -1:
                u[:, j] = (u[:, j] + u[:, j + separation]) % 2
                temp[j] = 0
                temp[j + separation] = 0
    return u


def transform_by_mat(u, factor_graph):
    """
    Input: u - same as above
    Output: x - same as above
    """
    return np.mod(np.matmul(u, factor_graph), 2)
