# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 08:30:10 2016

@author: jiang_y
"""

# Good tutorial: http://cs231n.github.io/python-numpy-tutorial/

# numpy: ndarray
# python: list
# pandas: dataframe, series

########### Create new variables ##############

### python data types ###

# list: ordered collection of objects of any types.
data = []
data.append(1)   # append new element
data.append('nice hat')

# list comprehesion
a = [1,2,3,4,2,3]
myList = [item*4 for item in a]
myList - [item*4 for item in a if item>2]

# array:
    
# Dictionary: unordered key-value pairs 
jj = {}    
jj['dog'] = 'dalmatian'
jj[1] = 42

    
### numpy data types ###

# array(numpy.ndarray)
from numpy import array
mm=array((1,1,1))
pp=array((3,4,5))
pp+mm
pp*mm
pp*2
pp**2
pp[1]

jj = array([[1,2,3],[1,1,1]])
jj[0]
jj[0][0]
jj[0,0]
jj[jj>1]

np.ones((2,3))
np.zeros((2,3))
np.eye(2)
np.random.random((2,2))

# matrix
from numpy import mat,matrix
ss = mat([1,2,3])
mm = matrix([1, 2, 3])
jj = mat([[1,2,3],[8,8,9]])
pyList = [5, 11, 1605]
mat(pyList)

jj[1,:]
jj[1,0:2]

shape(ss)
ss.shape
ss.sort()       # will sort ss and save in ss
ss.argsort()    # the indices of matrix if sort
ss.mean()

mm*ss.T

multiply(mm,ss)      #element-wise multiplication

########### Conversions #######################
a = df.as_matrix()

# series to array
a.values

# list to matrix
mat(data)


############# Data operations #################

