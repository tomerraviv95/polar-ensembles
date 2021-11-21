import numpy as np

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

    divident_int = divident.astype(int)
    divisor_len = len(key)
    divident_len = np.shape(divident)[1]
    # Number of bits to be XORed at a time.
    pick = divisor_len
    xor_mat = np.full((batch_size,divisor_len),divisor)
    # Slicing the divident to appropriate
    # length for particular step
    tmp = np.copy(divident_int[:,0 : pick])

    while pick < divident_len:
        print(tmp)
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

def crc_encode(data,key):
    '''
    Input: data - data to encode, np array (size = batch_size X word length)
           key - key to encode by, np array (size = 1 X crc_key)
    Output: codeword - encoded data, np array (size = batch_size X word length)
    '''
    l_key = key.size
    batch_size = np.shape(data)[0]
    # Appends n-1 zeroes at end of data
    appended_data = np.concatenate((data,np.zeros([batch_size,l_key-1])),axis=1)
    remainder = mod2div(appended_data, key)

    # Append remainder in the original data
    codeword = np.concatenate((data,remainder[:,1:]),axis=1) # first bit always 0
    print("Remainder : ", remainder)
    print("Encoded Data (Data + Remainder) : ",
          codeword)
    return codeword

def crc_check(data,key):
    '''
    Input: data - data to decode, np array (size = batch_size X word length)
           key - key to encode by, np array (size = 1 X crc_key)
    Output: crc_value - np array (size = batch_size X word length)
    '''
    crc_value = mod2div(data,key)
    return crc_value[:,1:]
