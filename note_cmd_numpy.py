# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 08:47:43 2017

@author: jiang_y
"""

# Ref: http://cs231n.github.io/python-numpy-tutorial/
import numpy as np

############################ Arrays ############################
# Construct an array
a = np.array([1, 2, 3])  # Create a rank 1 array
b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array

a = np.zeros((2,2))  # Create an array of all zeros
b = np.ones((1,2))   # Create an array of all ones
c = np.full((2,2), 7) # Create a constant array
d = np.eye(2)        # Create a 2x2 identity matrix
e = np.random.random((2,2)) # Create an array filled with random values

x = np.array([[1,2],[3,4]])
np.ones_like(x)

# Correct way to copy array. Ref: http://stackoverflow.com/questions/3059395/numpy-array-assignment-problem
b = a.copy()

######################## Array indexing ########################
# slicing
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]
# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a  
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]

# Integer array indexing                            
a = np.array([[1,2], [3, 4], [5, 6]])
# An example of integer array indexing.
# The returned array will have shape (3,) and 
print a[[0, 1, 2], [0, 1, 0]]  # Prints "[1 4 5]"

# useful trick
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])
print a[np.arange(4), b]  # Prints "[ 1  6  7 11]"
a[np.arange(4), b] += 10
print a

# Boolean array indexing
a = np.array([[1,2], [3, 4], [5, 6]])
bool_idx = (a > 2)  # Find the elements of a that are bigger than 2;
print bool_idx      # Prints "[[False False]
print a[bool_idx]  # Prints "[3 4 5 6]"
# We can do all of the above in a single concise statement:
print a[a > 2]     # Prints "[3 4 5 6]"

######################## Array math ########################
# basic math
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
print x + y
print np.add(x, y)
print x - y
print np.subtract(x, y)
print x * y
print np.multiply(x, y)
print x / y
print np.divide(x, y)
print np.sqrt(x)

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
v = np.array([9,10])
w = np.array([11, 12])
print v.dot(w)
print np.dot(v, w)
print x.dot(v)
print np.dot(x, v)
print x.dot(y)
print np.dot(x, y)

# Broadcasting
# add a constant vector to each row of a matrix.
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print y 