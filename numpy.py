import numpy as np

# Creating arrarys
v = np.array([1, 2, 3])

# Creating a 2D array
m = np.array([[1, 2, 3], [4, 5, 6]])

# Creating a 3D array
t = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(v)
print(m)
print(t)

# Convenient methods to create arrays initialized with specific values like zeros and ones

n_zero = np.zeros((3, 4))  # 3 rows and 4 columns
n_ones = np.ones((2, 3))   # 2 rows and 3 columns
n_range = np.arange(0, 10, 2)  # values from 0 to 10 with a step of 2

# Indexing  start from 0
print(v[0])
print(m[-2, 1])  # Accessing element in 2nd row and 2nd column : 5
print(t[1, 0, 1])  # Accessing element in 2nd block, 1st row and 2nd column : 6

# Slicing
print(v[1:3])  # elements from index 1 to 2 : [2 3]
print(m[:, 1:3])  # all rows and columns from index 1 to 2 : [[2 3] [5 6]]


q = np.array([1,2,3,4,5,6,7,8,9])
# boolean array indexing
print(q[q > 5])  # elements greater than 5 : [6 7 8 9]

# Numpy Operations
w = np.array([10, 20, 30, 40])
x = np.array([1, 2, 3, 4])
print(w + x)  # element-wise addition : [11 22 33 44]
print(w - x)  # element-wise subtraction : [ 9 18 27 36]
print(w * x)  # element-wise multiplication : [ 10  40  90 160]
print(w / x)  # element-wise division : [10. 10. 10. 10.]
print(w ** 2)  # element-wise exponentiation : [ 100  400 900 1600]

# Unary Operations
n = np.array([1, 2, 3, -4, -5])
res = np.absolute(n)  # absolute values : [1 2 3 4 5]
print(res) # Ans : [1 2 3 4 5]
print(np.mean(n))  # mean : -0.6 
print(np.median(n))  # median : 2.0
print(np.std(n))  # standard deviation : 3.415650255319866

# Binary Operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))  # dot product : 32
print(np.cross(a, b))  # cross product : [-3  6 -3]
print(np.add(a,b))  # sum of elements in a : [5 7 9]
print(np.subtract(b,a))  # difference of elements in b and a : [3 3 3]
print(np.multiply(a,b))  # element-wise multiplication : [ 4 10 18]
print(np.divide(b,a))  # element-wise division : [4.  2.5  2.]

# Numpy ufuncs
y = np.array([0, np.pi/2, np.pi]) # array of angles in radians
print(np.sin(y))  # sine values : [0. 1. 0.]

v = np.array([0, 30, 45, 60, 90]) # array of angles in degrees
res = np.exp(v) # ans : [1.00000000e+00 1.06864746e+13 3.49342711e+19 1.14200739e+26 1.22040329e+39]
print(res)  # exponential values
np.sqrt(v)  # square root values : [0.5.47722558 6.70820393 7.74596669 9.48683298]

# Sorting Arrays
type = [('name', 'S10'), ('grad_year', int), ('cgpa', float)]

students = np.array([('John', 2018, 8.5), ('Jane', 2019, 9.0), ('Dave', 2017, 8.0)], dtype=type) # values to be put in aaray
print(np.sort(students, order='cgpa'))  # sorting by cgpa
print(np.sort(students, order=['grad_year', 'cgpa']))  # sorting by graduation year

