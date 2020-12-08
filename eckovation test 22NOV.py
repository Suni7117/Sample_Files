#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from random import sample
import shutil
from zipfile import ZipFile
import warnings
warnings.filterwarnings("ignore") 


# In[ ]:





# In[6]:


my_dir = os.getcwd() 
zip_folder = os.path.join(my_dir,"DimensionalityReduction.zip.zip")
print("Path to the zipped folder is {}".format(zip_folder))
with ZipFile(zip_folder, 'r') as zip: 
    zip.extractall()


# In[7]:


import numpy as np
import os
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# In[22]:


folder = "Dataset"
images = []
print(os.listdir(folder))
for file in os.listdir(folder):
   
    img = mpimg.imread(os.path.join(folder, file))
    if img is not None:
        images.append(img)   
        a = ('MFCC_N.npy')                    


# In[38]:


folder = ['MFCC_N.npy','MFCC_S.npy']


# In[23]:


from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# calculate the mean of each column
M = mean(A.T, axis=1)
print(M)
# center columns by subtracting column means
C = A - M
print(C)
# calculate covariance matrix of centered matrix
V = cov(C.T)
print(V)
# eigendecomposition of covariance matrix
values, vectors = eig(V)
print(vectors)
print(values)
# project data
P = vectors.T.dot(C.T)
print(P.T)


# # 3 Compute the optimal direction vector m^ (unit vector) for LDA.

# In[24]:


import numpy as np
import matplotlib.pyplot as plt


# In[26]:


x1 = [[1,1],[2,2],[3,3],[5,4],[4,5],[6,6],[8,7]]
x2 = [[-2,3],[-1,4],[1,5],[3,6],[4,7],[2,8],[5,9]]


# In[27]:


x1 = np.array(x1)
x2 = np.array(x2)
print(x1.shape,x2.shape)


# In[28]:


plt.figure()
plt.title("Scatter Plot")
plt. xlabel("X-axis")
plt. ylabel("Y-axis")
plt.scatter(x1[:,0],x1[:,1],c='r')
plt.scatter(x2[:,0],x2[:,1],c='b')
plt.show()


# # 5.Plot the normalized histograms of zN and zS in two different colors (red and blue).
# 

# In[29]:


import matplotlib.pyplot as plt


# In[30]:


x = [21,22,23,4,5,6,77,8,9,10,31,32,33,44,35,36,37,39,49,50,100]
num_bins = 5
n,bins, patches = plt.hist(x, num_bins, facecolor = 'blue')
plt.show()


# In[31]:


x = [21,22,23,4,5,6,77,8,9,10,31,32,33,44,35,36,37,39,49,50,100]
num_bins = 5
n,bins, patches = plt.hist(x, num_bins, facecolor = 'red')
plt.show()


# # 4. Project the vector data in arrays N and S to generate the respective array of scalars zN and zS.

# In[1]:



import numpy as np

arr0 = np.array([1,2])

arr1 = np.array([[1,2],
                 [3,4]])

mat1 = np.matrix([[1,2],
                  [3,4]])
print("arr1 is:")
print(arr1)
print("mat1 is:")
print(mat1)

arr2 = np.array([[[1,2],
                  [3,4]],
                 [[5,6],
                  [7,8]]])

print("arr2 is:")
print(arr2)
print("Shape of arr0 is: {}".format(arr0.shape)) 
print("Size of arr0 is: {}".format(arr0.size))

print("Shape of arr1 is: {}".format(arr1.shape))
print("Size of arr1 is: {}".format(arr1.size))
print("Number of dimensions in arr1 is {}".format(arr1.ndim))  ## number of dimensions(or rank) of the numpy array
print("Datatype of arr1 is: {}".format(type(arr1))) ## datatype of arr1

print("\n")
print("Shape of arr2 is: {}".format(arr2.shape))


# In[ ]:




