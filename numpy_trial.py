import numpy as np
from numpy.testing._private.utils import jiffies

a = [1,2,3,4,5]
b = np.array([a])
print(b)
print(np.sum(b))

c = np.transpose(b)
print(c)

#a_i = np.array([1,2j,3j,-4j])
#b_i = np.conj(a_i)
#print(a_i)
#print(b_i)

