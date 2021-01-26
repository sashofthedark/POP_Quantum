import numpy as np
from numpy import conj
from math import sqrt

from numpy.core.fromnumeric import _argsort_dispatcher

class Vector():
    def __init__(self, vector, mult_factor):
        self.vector = vector
        #this should already be a numpy array
        self.factor = mult_factor

class Matrix():
    def __init__(self,matrix,mult_factor):
        #the matrix should be already a numpy array
        self.matrix = matrix
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

def IsHermitian(InputMatrix: Matrix):
    #checks whether the matrix is hermitian
    if len(InputMatrix.matrix[0,:]) != len(InputMatrix.matrix[:,0]):
        return False
        #if the matrix is not square, it is not Hermitian, and no further calculation is needed
    MatrixDagger = np.conj(np.transpose(InputMatrix.matrix))
    if  np.any(MatrixDagger - InputMatrix.matrix):
        return False
    else:
        return True

def IsPure(InputMatrix:Matrix):
    #checks whether the density matrix is pure or mixed
    #first, lets check it's a density matrix at all! - I will write a decorator for that
    SquaredMatrix = InputMatrix.matrix.dot(InputMatrix.matrix)
    if np.trace(SquaredMatrix) == np.trace(InputMatrix.matrix):
        return True
    else:
        return False

# Checking the functions

# vector1 = Vector(np.array([[1, 1]]), 1/sqrt(2))
# print(VectorNormalized(vector1))
# try:
#     vector11 = Vector(np.array([[1, 1]]), 1 / sqrt(2))
#     vector22 = Vector(np.array([[1, 1, 2]]), 1 / sqrt(2))
#     print(VectorsOrthogonal(vector11, vector22))
# except ValueError as V:
#     print(f'There was value error and its message is {V}')

# vector11 = Vector(np.array([[1, 1]]),1/sqrt(2))
# vector22 = Vector(np.array([[1, 1]]),1/sqrt(2))
# try:
#     VectorsSpan(vector11,vector22)
# except ValueError as V:
#     print(f'There was ValueError with the message {V}')

# DoTheySpan = VectorsSpan(vector11,vector22)
# print(DoTheySpan)

HermMatrix = np.array([[2,1+1j,2-1j],[1-1j,1,1j],[2+1j,-1j,1]])
NonHermMatrix = np.array([[1,2],[3,400]]) 
print(f'For Hermitian Matrix we get {IsHermitian(Matrix(HermMatrix,1))}')
print(f'For Non Hermitian Matrix we get {IsHermitian(Matrix(NonHermMatrix,1))}')