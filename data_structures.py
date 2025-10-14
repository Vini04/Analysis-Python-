# Lists
# Ordered, mutable collections of items, can be changes after creation, order in which elements are retrieved is guaranteed.
# Can store items of different data types.
# Defined using square brackets [].

my_list = [1, 2, 3, 'apple', 'banana', True]
print("List:", my_list)

# Accessing elements in a list : Indexing
print("First element of my_list:", my_list[0])

# multidimensional list (list of lists)
multi_list = [[1, 2, 3], ['apple', 'banana'], [True, False]]
print("Multidimensional List:", multi_list)

# Accessing  negative elements in a multidimensional list
print("Last element of second list in multi_list:", multi_list[1][-2]) # Accessing 'banana'

# Indexing and slicing
print("Sliced List (elements from index 1 to 3):", my_list[ 1:3]) # Slicing from index 1 to 3 (4 is exclusive)
# Answer: [2, 3]

# Adding elements to a list
my_list.append('orange') # adds 'orange' at the end of the list
print("List after appending 'orange':", my_list)

# Removing elements from a list
my_list.remove('banana') # removes 'banana' from the list
print("List after removing 'banana':", my_list)

# Sorting a list
num_list = [5, 2, 9, 1, 5, 6]
num_list.sort() # sorts the list in ascending order
print("Sorted num_list:", num_list)

# List comprehension
squared_list = [x**2 for x in range(5)]  # creates a list with squares of numbers from 0 to 4
print("List using comprehension (squared_list):", squared_list)





# Tuple
# Ordered, immutable collections of items.
# Similar to lists but their content cannot be changed after creation.
# Defined using parentheses ().

my_tuple = (1, 2, 3, 'apple', 'banana', False)
print("Tuple:", my_tuple)

# Creating tuple with help of list
my_tuple1 = (1,2, [3,4,5 , 'vineet']) # tuple with list inside
print("Tuple with list inside:", my_tuple1)

my_list2 = [1,2, (3,4,5 , 'vineet')] # list with tuple inside
tuple_from_list = tuple(my_list2) # converting list to tuple
print("Tuple created from list:", tuple_from_list)

# Accessing elements in a tuple
print("First Element :", my_tuple(my_tuple[0]))

# Accessing negative elements in a tuple
print("Last Element :", my_tuple1[-2]) # Accessing list inside the tuple

# Unpacking a tuple
a, b, c = my_tuple1
print("Unpacked values from my_tuple1:", a, b, c)
# Answer: 1 2 [3, 4, 5, 'vineet']




# Dictionaries 
# Unordered collections of key-value pairs, order in which elements are retrieved is not guaranteed.
# Each key must be unique and immutable, while values can be of any type and mutable.
# Defined using curly braces {} with key-value pairs separated by colons.

my_dict = {'name': 'Vineet', 'age': 24, 'is_student': False}
print("Dictionary:", my_dict)

# Accessing elements in a dictionary using key
print("Name from my_dict:", my_dict['name'])

# Accessing elements using get() method
print("Age from my_dict using get():", my_dict.get('age'))

# Creation of nested dictionary
nested_dict = { 'person1': {'name': 'Alice', 'age': 30}, 'person2': {'name': 'Bob', 'age': 25}}
print("Nested Dictionary:", nested_dict)

# Dictionary comprehension
squared_dict = {x: x**2 for x in range(5)}  # creates a dictionary with numbers as keys and their squares as values
print("Dictionary using comprehension:", squared_dict)

# Adding/Updating key-value pairs in a dictionary
my_dict['city'] = 'New York'  # adds a new key-value pair
print("Updated my_dict:", my_dict) # to show updated dictionary

# looping through dictionary
print("Keys in my_dict:") # to show keys in dictionary
for key in my_dict:
    print(key, end=' ') # end is used to print in same line with space
print() # for new line



# Sets
# Unordered collections of unique elements.
# Useful for tasks like removing duplicates or testing membership.
# Defined using curly braces {} or the set() constructor.

my_set = {1, 2, 3, 1, 'apple', 'banana'}
print("Set:", my_set)

#Accessing elements in a set using for loop
print("Elements in my_set:")
for item in my_set:
    print(item, end=' ') # end is used to print in same line with space
print()

# Checking element using in keyword
print("Is 'apple' in my_set?", 'apple' in my_set)

# Adding elements to a set
my_set.add('orange') # adds 'orange' to the set
print("Set after adding 'orange':", my_set)

# Removing elements from a set
my_set.remove(2) # removes 2 from the set
print("Set after removing 2:", my_set)

# Set operations
# Union
set_a = {1, 2, 3}
set_b = {3, 4, 5}
set_union = set_a.union(set_b) # or set_a | set_b
print("Union of set_a and set_b:", set_union)

# Intersection
set_intersection = set_a.intersection(set_b) # or set_a & set_b
print("Intersection of set_a and set_b:", set_intersection)

# Difference
set_difference = set_a.difference(set_b) # or set_a - set_b
print("Difference of set_a and set_b (set_a - set_b):", set_difference)

# Checking membership
print("Is 4 in set_b?", 4 in set_b) # True
print("Is 1 in set_b?", 1 in set_b) # False