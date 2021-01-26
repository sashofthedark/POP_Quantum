import numpy as np
from numpy.testing._private.utils import jiffies

# a = [1,2,3,4,5]
# b = np.array([a])
# print(b)
# print(np.sum(b))

# c = np.transpose(b)
# print(c)

#a_i = np.array([1,2j,3j,-4j])
#b_i = np.conj(a_i)
#print(a_i)
#print(b_i)

# v1 = np.array([[1,2]])
# v2 = np.array([2,4])
# cross_product = np.cross(v1,v2)
# print(cross_product)

# newrow = [1,2,3]
# A = np.array([[1,3,4]])
# A = np.vstack([A, newrow])
# for row in A:
#     print(row)

# Matrix1 = np.array([[1,2,3],[2,4,5]])
# print(len(Matrix1[0,:]))
# print(len(Matrix1[:,0]))

matrix1 = np.array([[1,2],[2,1]])
#print(matrix1)
matrix2 = np.array([[1,0],[1,3]])
#print(matrix2)
multiplied = matrix1.dot(matrix2)
print(multiplied)
print(np.trace(multiplied))