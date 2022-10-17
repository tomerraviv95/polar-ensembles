import numpy as np

def get_crc_key(order):
    key = np.zeros(order)
    indices = []
    if order == 11:
        indices = [1,5,9,10,11]
    elif order == 16:
        indices = [1,5,12,16]

    for i in indices:
        key[i] = 1

    return key


def mod2div(divident, divisor):

    # Number of bits to be XORed at a time.
    pick = len(divisor)

    # Slicing the divident to appropriate
    # length for particular step
    tmp = np.copy(divident[0 : pick])

    while pick < len(divident):
        if tmp[0] == 1:

            # replace the divident by the result
            # of XOR and pull 1 bit down
            tmp[:-1] = np.bitwise_xor(divisor[1:], tmp[1:])
            tmp[-1] = divident[pick]

        else:   # If leftmost bit is '0'
            # If the leftmost bit of the dividend (or the
            # part used in each step) is 0, the step cannot
            # use the regular divisor; we need to use an
            # all-0s divisor.
            tmp[:-1] = tmp[1::]
            tmp[-1] = divident[pick]
        # increment pick to move further
        pick += 1

    # For the last n bits, we have to carry it out
    # normally as increased value of pick will cause
    # Index Out of Bounds.
    if tmp[0] == 1:
        tmp = np.bitwise_xor(divisor, tmp)

    reminder = tmp
    return reminder

def crc_encode(data,key):

    l_key = key.size
    # Appends n-1 zeroes at end of data
    appended_data = np.append(data,np.zeros(l_key-1))
    remainder = mod2div(appended_data, key)

    # Append remainder in the original data
    codeword = np.append(data,remainder[1:]) # first bit always 0
    print("Remainder : ", remainder)
    print("Encoded Data (Data + Remainder) : ",
          codeword)
    return codeword

def crc_check(data,key):
    rem = mod2div(data,key)
    return rem[1:]

if __name__ == "__main__":
    batch_size = 2
    key = [1,1,0,1]
    a = np.array([1,0,0,1,0,1,0,0,0])
    rem = mod2div(a,key)
    print(rem)




#     batch_size = 2
# key = np.array([1,1,0,1])
# a = np.array([[1,0,0,1,0,0],[1,0,0,1,0,1]])
# #rem = mod2div(a,key,batch_size)
#
# codeword = crc_encode(a,key)
# print("codeword",end="")
# print(codeword)
# check = crc_check(codeword,key)
# print("check",end="")
# print(check)