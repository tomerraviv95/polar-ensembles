import numpy as np
from numpy import matlib as mb
import torch

def get_crc_key(order=11):
    '''
    Input: order(int) - order of the crc polynomial (11 or 16)
    Output: key - the crc key, highest order is lowest index
    '''
    key = np.zeros(order+1)
    indices = []
    if order == 11:
        indices = [0,5,9,10,11]
    elif order == 16:
        indices = [0,5,12,16]

    for i in indices:
        key[order-i] = 1

    return key


def mod2div(divident, divisor):
    '''
    Input: divident - np array (size = batch_size X word length)
           divisor - np array (size = 1 X crc_key)
    Output: reminder - reminder of divident/divisor of size divisor
    '''
    batch_size = np.shape(divident)[0]
    if isinstance(divident, torch.Tensor):
        divident = divident.detach().numpy()
    divident_int = divident.astype(int)
    divisor_int = divisor.astype(int)
    divisor_len = len(divisor_int)
    divident_len = np.shape(divident)[1]
    # Number of bits to be XORed at a time.
    pick = divisor_len
    xor_mat = np.full((batch_size,divisor_len),divisor_int)
    # Slicing the divident to appropriate
    # length for particular step
    tmp = np.copy(divident_int[:,0 : pick])

    while pick < divident_len:
        extended_tmp = np.full((batch_size,divisor_len),np.reshape(tmp[:,0],(batch_size,1)))
        word_wise_divisor = extended_tmp*xor_mat
        tmp[:,:-1] = np.bitwise_xor(word_wise_divisor[:,1:], tmp[:,1:])
        tmp[:,-1] = divident_int[:,pick]
        pick += 1

    # For the last n bits, we have to carry it out
    # normally as increased value of pick will cause
    # Index Out of Bounds.
    extended_tmp = np.full((batch_size,divisor_len),np.reshape(tmp[:,0],(batch_size,1)))
    word_wise_divisor = extended_tmp*xor_mat
    reminder_int = np.bitwise_xor(word_wise_divisor, tmp)

    reminder = reminder_int.astype(float)
    return reminder

def crc_encode(data,order):
    '''
    Input: data - data to encode, np array (size = batch_size X word length)
           order - the polynomial order for key generation (int)
    Output: codeword - encoded data, np array (size = batch_size X word length)
    '''
    key = get_crc_key(order)
    l_key = key.size
    batch_size = np.shape(data)[0]
    # Appends n-1 zeroes at end of data
    appended_data = np.concatenate((data,np.zeros([batch_size,l_key-1])),axis=1)
    remainder = mod2div(appended_data, key)

    # Append remainder in the original data
    codeword = np.concatenate((data,remainder[:,1:]),axis=1) # first bit always 0
    # print("Remainder : ", remainder)
    # print("Encoded Data (Data + Remainder) : ",codeword)
    return codeword

def crc_check(data,order):
    '''
    Input: data - data to decode, np array (size = batch_size X word length)
           order - the polynomial order for key generation (int)
    Output: crc_value - np array (size = batch_size X word length)
    '''
    key = get_crc_key(order)
    crc_value = mod2div(data,key)
    return crc_value[:,1:]

def crc2int(crc : np.array):
    batch_size = np.shape(crc)[0]
    crc_val = np.zeros((batch_size,1)).astype(int)
    for row in range(batch_size):
        crc_val[row] = int("".join(str(int(x)) for x in crc[row]),2)
    return crc_val.flatten('F')

def addBin(arr, val):
    ''' Binary addition of val to each element of arr
        arr dim is: (idx,binary values)'''
    to_tensor = False
    if isinstance(arr,torch.Tensor):
        arr = arr.cpu().detach().numpy()
        to_tensor = True
    val = np.array([int(i) for i in np.binary_repr(val)])
    size_arr = np.shape(arr)
    val = np.pad(val,pad_width=((size_arr[1]-len(val)),0))
    val = mb.repmat(val,size_arr[0],1)
    c_in = np.array(([0]*size_arr[0]))
    res = np.zeros(size_arr)
    for i in reversed(range(size_arr[1])):
        A = arr[:,i].astype(int)
        B = val[:,i]
        res[:,i] = np.bitwise_xor(np.bitwise_xor(A,B), c_in)
        a = np.bitwise_and(A,B)
        b = np.bitwise_and(A,c_in)
        c = np.bitwise_and(c_in,B)
        c_in = np.bitwise_or(a,b)
        c_in = np.bitwise_or(c_in,c)
    c_in = np.reshape(c_in,(len(c_in),1))
    res = np.concatenate((c_in,res),axis=1)
    if to_tensor:
        res = torch.Tensor(res)
    return res

if __name__ == "__main__":
    arr = np.array([[1,0,1],[0,0,1]])
    val = 3
    res = addBin(arr,val)
    print(res)