# NumPy
# NumPy is a Python library.
#
# NumPy is used for working with arrays.
#
# NumPy is short for "Numerical Python".
import axis as axis
import numpy
import numpy as np
from numpy import random
import numpy as geek

arr = numpy.array([1, 2, 3, 4, 5])
print(arr)
print(np.__version__)
print("")
#
# Creating Arrays
arr = np.array([1, 3, 4, 5, 6, 6])
print(arr)
print(type(arr))
print("")

# Dimensions in Arrays
# 0-D Arrays
arr = np.array(43)
print(arr)

# 1-D Arrays
arr = np.array([1, 2, 3, 4, 556, 6])
print(arr)
print("")

# 2-D Arrays
arr = np.array([[1, 2, 4], [23, 5, 7]])
print(arr)

# 3-D arrays
arr = np.array([[[1, 2, 4], [3, 5, 6], [4, 5, 6, ], [65, 64, 43], [8, 8, 9]]])
print(arr)
# Dimensions
a = np.array(43)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [3, 4, 5]])
d = np.array([[[1, 3, 5], [3, 4, 5]], [[1, 2, 3], [4, 5, 6]]])
print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)
print("")

# Higher Dimensional
arr = np.array([1, 2, 4, 5], ndmin=4)
print(arr)
print('number of dimensions:', arr.ndim)
print("")

# Array Indexing
arr = np.array([1, 2, 3, 5])
print(arr[1])
print(arr[0])
print(arr[2] + arr[3])
print("")

# 2-D Arrays
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])
print('2nd element on 1 st row:', arr[0, 1])
print("")

# 3-D Arrays
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr[0, 1, 2])
print("")

# Array Slicing
arr = np.array([1, 2, 3, 4, 5, 6, 7])
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
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[1, 1:4])
print("")
print(arr[0:4, 4])
print(arr[0:2, 3])
print(arr[0:2, 1:4])
print("")

# Data Types in Python

# strings - used to represent text data, the text is given under quote marks. e.g. "ABCD"
# integer - used to represent integer numbers. e.g. -1, -2, -3
# float - used to represent real numbers. e.g. 1.2, 42.42
# boolean - used to represent True or False.
# complex - used to represent complex numbers. e.g. 1.0 + 2.0j, 1.5 + 2.5j

arr = np.array([1, 2, 3, 4])
print(arr.dtype)
print("")
arr = np.array(['apple', 'banana', 'cherry'])
print(arr.dtype)
print("")
arr = np.array([1, 2, 3, 4], dtype='S')
arr1 = np.array([1, 2, 3, 4], dtype='i4')
print(arr)
print(arr.dtype)
print("")

# NumPy Array Copy vs View
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[1] = 35
print(arr)
print(x)
print("")

# VIEW:
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[1] = 35
print(arr)
print(x)
print("")

# NumPy Array Shape
# Reshape From 1-D to 2-D
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr.shape)
print("")
# NumPy Array Reshaping
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(4, 3)
print(newarr)
print("")

# Reshape From 1-D to 3-D
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(2, 3, 2)
print(newarr)
print("")

# NumPy Array Iterating
arr = np.array([1, 2, 3])
for x in arr:
    print(x)
print("")

# Iterating 2-D Arrays
print("2D Array-")
arr = np.array([[1, 2, 3], [4, 5, 6]])
for x in arr:
    print(x)
print("")

# Iterating 3-D Arrays
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
for x in arr:
    print(x)
print("")

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
for x in arr:
    for y in x:
        for z in y:
            print(z)
print("")

# NumPy Joining Array
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)
print("")

print("2d array")
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr = np.concatenate((arr1, arr2), axis=1)
# joining array as stack function

arr = np.stack((arr1, arr2), axis=1)
print(arr)
print("")

# Searching Arrays
arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print(x)
print("")

# Sorting Arrays
arr = np.array([3, 2, 0, 1])
print(np.sort(arr))
print("")
arr = np.array(['banana', 'cherry', 'apple'])
print(np.sort(arr))
print("")

# Sorting a 2-D Array
arr = np.array([[3, 2, 4], [5, 0, 1]])
print(np.sort(arr))
print("")

# Filtering Arrays
arr = np.array([41, 42, 43, 44])
x = [True, False, True, False]
newarr = arr[x]
print(newarr)
print("")

arr = np.array([41, 42, 43, 44])
filter_arr = []
for element in arr:
    if element > 42:
        filter_arr.append(True)
    else:
        filter_arr.append(False)
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)
print("")

print("2 method")
arr = np.array([1, 2, 3, 4, 5, 6, 7])
filter_arr = []
for element in arr:
    if element % 2 == 0:
        filter_arr.append(True)
    else:
        filter_arr.append(False)
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)
print("")

arr = np.array([1, 2, 3, 4, 5, 6, 7])
filter_arr = arr % 2 == 0
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

# NumPy Random
x = random.rand()
print(x)
print("")

# Generate Random Number
x = random.randint(100)
print(x)
print("")

arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Array is of type:", type(arr))
print("no of dimensions:", arr.ndim)
print("shape of array:", arr.shape)
print("size of array:", arr.size)
print("Array stores element of type:", arr.dtype)
print("")

# Basic Slicing and Advanced Indexing in NumPy Python
list1 = [1, 2, 3, 4, 5, 6]
list2 = [10, 9, 8, 7, 6, 5]
print(list1 + list2)
print("")

list1 = [1, 2, 3, 4, 5, 6]
list2 = [10, 9, 8, 7, 6, 5]
a1 = np.array(list1)
a2 = np.array(list2)
print(a1 * a2)
print("")

# Numpy | Data Type Objects
a = np.array([1])
print("type is:", type(a))
print("dtype is:", a.dtype)
print("")

dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, 2)])
x = np.array([('ashish', (8.0, 9.0)), ('jhon', (6.0, 9.0))], dtype=dt)
print(x[1])
print("Grades of jhon are:", x[1]['grades'])
print("names are:", x['name'])
print("")

# Numpy | Iterating Over Array
a = geek.arange(12)
a = a.reshape(3, 4)
print(a)
print()
print('Modified array is :')
for x in geek.nditer(a):
    print(x)
print("")

a = geek.arange(12)
a = a.reshape(3, 4)
print('first array is:')
print(a)
print()
print('Second array is :')
b = geek.array([5, 6, 7, 8], dtype=int)
print(b)
print()
print('Modified array is :')
for x, y in geek.nditer([a, b]):
    print("%d:%d" % (x, y))
print("")

# Numpy | Binary Operations
in_num1 = 10
in_num2 = 11
print("Input number1:", in_num1)
print("Input number2:", in_num2)
out_num = geek.bitwise_and(in_num1, in_num2)
print("bitwise_and of 10 and 11:", out_num)
print("")

in_num = 10
print("Input number:", in_num)
out_num = geek.invert(in_num)
print("inversion of 10:", out_num)
print("")

in_num = 10
print("Input number:", in_num)
out_num = geek.binary_repr(in_num)
print("binary representation of 10:", out_num)
print("")

a = np.array([[[1, 0, 1], [0, 1, 0]], [[1, 1, 0], [0, 0, 1]]])
b = np.packbits(a, axis=-1)
print(b)
print("")

a = np.array([[2], [7], [23]], dtype=np.uint8)
b = np.unpackbits(a, axis=1)
print(b)
print("")

a = np.array(['ashish', 'mundra'])
print(np.char.count(a, 'ashish'))
print(np.char.count(a, 'mun'))
print("")

# a=np.char.unequal('ash','ish`')
# print(a)

# Sorting
a = np.array([[12, 15], [10, 1]])
arr1 = np.sort(a, axis=0)
print("along first axis:\n", arr1)

a = np.array([[10, 15], [12, 1]])
arr2 = np.sort(a, axis=-1)
print("\n along first axis:\n", arr2)

a = np.array([[12, 15], [10, 11]])
arr1 = np.sort(a, axis=None)
print("\n Along none axis:\n", arr1)
print("")

a = np.array([9, 3, 1, 3, 5, 7, 8])
b = np.array([4, 3, 5, 7, 8, 9, 9])
print('column a,column b')
for (i, j) in zip(a, b):
    print(i, '', j)
ind = np.lexsort((b, a))
print('sorted indices-->', ind)
print("")

# Searching
array = geek.arange(12).reshape(3, 4)
print("INPUT ARRAY:\n", array)
print("\nMax element :", geek.argmax(array))
print(("\nIndices of max element :", geek.argmax(array, axis=0)))
print(("\nIndices of max element :", geek.argmax(array, axis=1)))
print("")

# Counting
a = np.count_nonzero([[0, 1, 2, 3, 4], [4, 6, 7, 8, 5]])
b = np.count_nonzero(([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]]
                      , axis == 0))
print("Number of nonzero values is :", a)
print("2. Number of nonzero values is :", b)
print("")

# randint()
# function
out_arr = geek.random.randint(low=4, size=(2, 3))
print("output 2d array filled with random intergers:", out_arr)
print("")
print("example 2")
out_arr = geek.random.randint(2, 10, (2, 3, 4))
print("output 3d array filled with random interger :", out_arr)
print("")

# random_sample()
out_val = geek.random.random_sample()
print("output random float value:", out_val)
print("")

out_arr = geek.random.random_sample((3, 2, 1))
print("output 3d array filled  with random floats :", out_arr)
print("")

# ranf() function
print("1 d array")
out_val = geek.random.ranf()
print("output random  float value:", out_val)
print("")
print("2 d array")
out_arr = geek.random.ranf(size=(2, 1))
print("output 2d array  filled with random floats:", out_arr)
print("")
print("3d array")
out_arr = geek.random.ranf((3, 3, 2))
print("output 3d array filled with random floats:", out_arr)
print("")

# random_integers()
out_arr = geek.random.randint(low=0, high=5, size=4)
print("ouput 1 d array filled with random interger :", out_arr)
print("")
