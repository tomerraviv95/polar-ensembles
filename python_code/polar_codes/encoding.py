import numpy as np

def encode(target, crc_gm, code_len, info_ind, crc_ind, system_enc: bool = False, factor_graph: np.array = None):
    """
    Input: u - Information matrix with Batch x info (each row is information bits), G - Generator matrix with info x N
    Output: x - Encoded matrix Batch x code_len
    """
    len_crc = sum(crc_ind)
    batch_size, _ = target.shape
    u = np.zeros((batch_size, code_len))
    if len_crc:
        target_crc = (np.matmul(target, crc_gm.T) % 2)
        u[:, info_ind] = target_crc[:, len_crc:]
        u[:, crc_ind] = target_crc[:, :len_crc]
    else:
        u[:, info_ind] = target

    # bit reversal
    # bitreversedindices = np.zeros(code_len, dtype=int)
    # for bit in range(code_len):
    #     b = '{:0{width}b}'.format(bit, width=int(np.log2(code_len)))
    #     bitreversedindices[bit] = int(b[::-1], 2)
    # u = u[:, bitreversedindices]

    x = transform_by_mat(u, factor_graph)
    if system_enc:  # TODO: check
        raise NotImplementedError
        # print("systematic")
        # info_bits = np.logical_or(info_ind, crc_ind)
        # x[:, ~info_bits] = 0
        # x = transform_by_recursion(x)

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
