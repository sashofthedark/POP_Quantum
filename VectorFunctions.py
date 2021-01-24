import numpy as np
from numpy import conj
from math import sqrt

from numpy.core.fromnumeric import _argsort_dispatcher

class Vector():
    def __init__(self, vector, mult_factor):
        self.vector = np.array([vector])
        self.factor = mult_factor


def VectorNormalized(input_vector_user: Vector):
    # checks whether a vector is normalized
    # returns 1 if normalized, and normalization constant otherwise
    input_vector = input_vector_user.vector * input_vector_user.factor
    # multiplying the vector and the constant factor and putting it in input_vector
    input_vector_conj = np.transpose(np.conj(input_vector))
    normalization = np.sum(input_vector.dot(input_vector_conj))
    try:
        return 1 / sqrt(normalization)
    except ZeroDivisionError:
        return 0


def VectorsOrthogonal(input_vector_user1: Vector, input_vector_user2: Vector):
    # checks whether vectors are orthogonal - returns True if so, and False otherwise
    # currently also returns False if the vectors are not the same length
    v1 = input_vector_user1.vector
    v2 = input_vector_user2.vector
    v2_conj = np.conj(v2)
    if len(v1[0,:]) != len(v2[0,:]):
        raise ValueError(f'Vectors are not the same length: length of first vector is {len(v1)}'
                         f'and length of second vectors is {len(v2)}')
    else:
        multiplication = np.sum(v1.dot(np.transpose(v2_conj)))
        if abs(multiplication) < 1e-10:
            return True
        else:
            return False

#STILL WORK IN PROGRESS
def VectorsSpan(*args):
    #check whether all of the vectors are the same length and the number of vectors is equal to that length,
    #otherwise throw exceptions (different ones for each one of the cases).
    NumOfVectors = len(args)
    MatrixOfVectors = np.empty((NumOfVectors,NumOfVectors),dtype = np.complex)
    for i,item in enumerate(args):
        if len(item.vector[0,:])!= NumOfVectors:
            raise ValueError('One or more of the vectors is the wrong length')
        else:
            #MatrixOfVectors = np.array([item.vector[0,:]])
            MatrixOfVectors[i,:] = item.vector[0,:]
            #append a row to a matrix of vectors
    #now check whether any of the vectors are multiples of one another.
    for firstRow in range(NumOfVectors):
        for secondRow in range(firstRow+1,NumOfVectors):
            if np.cross(MatrixOfVectors[firstRow,:],MatrixOfVectors[secondRow,:]) == 0:
                return False
    return True

# Checking the functions

# vector1 = Vector([1, 1], 1/sqrt(2))
# print(VectorNormalized(vector1))
# try:
#     vector11 = Vector([1, 1], 1 / sqrt(2))
#     vector22 = Vector([1, 1, 2], 1 / sqrt(2))
#     print(VectorsOrthogonal(vector11, vector22))
# except ValueError as V:
#     print(f'There was value error and its message is {V}')

vector11 = Vector([1,1],1/sqrt(2))
vector22 = Vector([1,1],1/sqrt(2))
# try:
#     VectorsSpan(vector11,vector22)
# except ValueError as V:
#     print(f'There was ValueError with the message {V}')

DoTheySpan = VectorsSpan(vector11,vector22)
print(DoTheySpan)