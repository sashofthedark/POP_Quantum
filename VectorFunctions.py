from numpy import conj
from math import sqrt

from numpy.lib.index_tricks import _fill_diagonal_dispatcher

class Vector():
    def __init__(self,vector,mult_factor):
        self.vector = vector
        self.factor = mult_factor

def VectorNormalized(input_vector_user:Vector):
    #checks whether a vector is normalized
    #returns 1 if normalized, and normalization constant otherwise
    input_vector = [x*input_vector_user.factor for x in input_vector_user.vector]
    #multiplying the vector and the constant factor and putting it in input_vector
    normalization = sum([x*conj(x) for x in input_vector])
    try:
        return 1/sqrt(normalization)
    except ZeroDivisionError:
        return 0

def VectorsOrthogonal(input_vector_user1:Vector,input_vector_user2:Vector):
    #checks whether vectors are orthogonal - returns True if so, and False otherwise
    #currently also returns False if the vectors are not the same length
    v1 = input_vector_user1.vector
    v2 = input_vector_user2.vector
    if len(v1) != len(v2):
        raise ValueError(f'Vectors are not the same length: length of first vector is {len(v1)}'
                           f'and length of second vectors is {len(v2)}')
    else:
        multiplication = sum([a*conj(b) for a,b in zip(v1,v2)])
        if abs(multiplication) < 1e-10:
            return True
        else:
            return False

#def SpanSpace():

#Checking the functions

vector1 = Vector([1,1],1/sqrt(2))
print(VectorNormalized(vector1))

vector11 = Vector([1,1,3],1/sqrt(2))
vector22 = Vector([1,-1],1/sqrt(2))
print(VectorsOrthogonal(vector11,vector22))