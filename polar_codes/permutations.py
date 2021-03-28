import numpy as np
from itertools import permutations
from polar_codes.initialization import initialize_connections

## UNTESTED - RUN WITH CARE

def create_polar_permutations(code_len, num_of_perm_layers=None, num_of_permutations=None):
    """
    code_len: codeword length
    num_of_permutations: number of permutations
    num_of_perm_layers: number of permutation layers
    :return:
    pi_mat: bit permutation matrix size [Mxlog2(N)]
    """
    stages = int(np.log2(code_len))
    connections = initialize_connections(code_len).astype(np.uint8)
    if num_of_perm_layers is None:
        num_of_perm_layers = stages
    if num_of_permutations is None:
        num_of_permutations = np.math.factorial(num_of_perm_layers)
    pi_mat = []
    integers_array = list(range(stages))
    perms = permutations(integers_array[-num_of_perm_layers:])
    for p in perms:
        pi_mat.append(integers_array[:-num_of_perm_layers] + list(p))
    return polar_layer_to_index_perm(pi_mat[:num_of_permutations], connections=connections.T)


def polar_layer_to_index_perm(layer_perm, connections):
    index_perm = []
    for perm in layer_perm:
        if connections.shape[1] <= 8:
            c = np.pad(np.flip(connections[:, perm]), ((0, 0), (8 - connections.shape[1], 0)), 'constant')
            index_perm.append(np.packbits(c, axis=-1).flatten())
        elif connections.shape[1] <= 16:
            c = np.pad(np.flip(connections[:, perm]), ((0, 0), (16 - connections.shape[1], 0)), 'constant')
            index_perm.append(np.packbits(c.reshape(-1, 2, 8)[:, ::-1]).view(np.uint16))
        else:
            raise NotImplementedError
    return np.array(index_perm)

if __name__=="__main__":
    code_len = 64
    perm_group = create_polar_permutations(code_len, num_of_permutations=4).astype(np.int16)
    print(perm_group)
