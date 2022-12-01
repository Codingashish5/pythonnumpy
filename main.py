# NumPy
# NumPy is a Python library.
#
# NumPy is used for working with arrays.
#
# NumPy is short for "Numerical Python".
import numpy
import numpy as np
arr=numpy.array([1,2,3,4,5])
print(arr)
print(np.__version__)
print("")
#
# Creating Arrays
arr=np.array([1,3,4,5,6,6])
print(arr)
print(type(arr))
print("")


# Dimensions in Arrays
# 0-D Arrays
arr=np.array(43)
print(arr)


# 1-D Arrays
arr=np.array([1,2,3,4,556,6])
print(arr)
print("")

# 2-D Arrays
arr=np.array([[1,2,4],[23,5,7]])
print(arr)

# 3-D arrays
arr=np.array([[[1,2,4],[3,5,6],[4,5,6,],[65,64,43],[8,8,9]]])
print(arr)
# Dimensions
a=np.array(43)
b=np.array([1,2,3,4,5])
c=np.array([[1,2,3],[3,4,5]])
d=np.array([[[1,3,5],[3,4,5]],[[1,2,3],[4,5,6]]])
print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)
print("")

# Higher Dimensional
arr=np.array([1,2,4,5],ndmin=4)
print(arr)
print('number of dimensions:',arr.ndim)
print("")

# Array Indexing
arr=np.array([1,2,3,5])
print(arr[1])
print(arr[0])
print(arr[2]+arr[3])
print("")

# 2-D Arrays
arr=np.array([[1,2,3,4,5],[6,7,8,9,0]])
print('2nd element on 1 st row:',arr[0,1])
print("")

# 3-D Arrays
arr=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(arr[0,1,2])
print("")



# Array Slicing
arr=np.array([1,2,3,4,5,6,7])
print(arr[1:5])
print(arr[4:])
print(arr[4])
# Negative Slicing
print(arr[-3:-1])
# STEP
print(arr[1:5:2])
print(arr[::4])
print("")



# Slicing 2-D Arrays
arr=np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(arr[1,1:4])
print("")
print(arr[0:4,4])
print(arr[0:2,3])
print(arr[0:2,1:4])
print("")


# Data Types in Python

# strings - used to represent text data, the text is given under quote marks. e.g. "ABCD"
# integer - used to represent integer numbers. e.g. -1, -2, -3
# float - used to represent real numbers. e.g. 1.2, 42.42
# boolean - used to represent True or False.
# complex - used to represent complex numbers. e.g. 1.0 + 2.0j, 1.5 + 2.5j

arr=np.array([1,2,3,4])
print(arr.dtype)
print("")
arr=np.array(['apple','banana','cherry'])
print(arr.dtype)
print("")
arr=np.array([1,2,3,4], dtype='S')
arr=np.array([1,2,3,4], dtype='i4')
print(arr)
print(arr.dtype)
print("")


# NumPy Array Copy vs View
arr=np.array([1,2,3,4,5])
x=arr.copy()
arr[1]=35
print(arr)
print(x)
print("")

# VIEW:
arr=np.array([1,2,3,4,5])
x=arr.view()
arr[1]=35
print(arr)
print(x)
print("")


# NumPy Array Shape
# Reshape From 1-D to 2-D
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr.shape)
print("")
# NumPy Array Reshaping
arr= np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newarr=arr.reshape(4,3)
print(newarr)
print("")

# Reshape From 1-D to 3-D
arr=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newarr=arr.reshape(2,3,2)
print(newarr)
print("")

# NumPy Array Iterating
arr=np.array([1,2,3])
for x in arr:
    print(x)
print("")

# Iterating 2-D Arrays
print("2D Array-")
arr=np.array([[1,2,3],[4,5,6]])
for x in arr:
    print(x)
print("")

# Iterating 3-D Arrays
arr=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
for x in arr:
    print(x)
print("")


arr=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
for x in arr:
    for y in x:
        for z in y:
            print(z)
print("")

# NumPy Joining Array
arr1=np.array([1,2,3])
arr2=np.array([4,5,6])
arr=np.concatenate((arr1,arr2))
print(arr)
print("")

print("2d array")
arr1=np.array([[1,2],[3,4]])
arr2=np.array([[5,6],[7,8]])
arr=np.concatenate((arr1,arr2),axis=1)
# joining array as stack function

arr = np.stack((arr1, arr2), axis=1)
print(arr)
print("")

# Searching Arrays
arr=np.array([1,2,3,4,5,4,4])
x=np.where(arr==4)
print(x)
print("")

# Sorting Arrays
arr=np.array([3,2,0,1])
print(np.sort(arr))
print("")
arr=np.array(['banana','cherry','apple'])
print(np.sort(arr))
print("")

# Sorting a 2-D Array
arr=np.array([[3,2,4],[5,0,1]])
print(np.sort(arr))