 # open() function
file_path = 'example.txt'
file = open(file_path, 'mode') # modes can be 'r', 'w', 'a', etc.
print(file)

# close() function
file = open(file_path, 'mode') # opens the file
# perform file operations here
file.close() # closes the file
print(file)

# read() function
file = open(file_path, 'r') # opens the file in read mode   
content = file.read() # reads the entire content of the file
print("File Content:\n", content)
file.close() # closes the file

# write() function
file = open('output.txt', 'w') # opens (or creates) the file in write mode
file.write("Hello, World!\n") # writes a string to the file
print("Data written to file.")
file.close() # closes the file

# File properties
file = open(file_path, 'r') # opens the file in read mode
print("File Name:", file.name) # prints the name of the file
print("File Mode:", file.mode) # prints the mode in which the file is opened
print("Is File Closed?:", file.closed) # checks if the file is closed

file.close() # closes the file
print("Is File Closed after closing?:", file.closed) # checks if the file is closed

# with statement
with open("file_name.txt", 'mode') as file:
    # perform file operations here
   content = file.read() # reads the entire content of the file
   print(content)

# handling exceptions while closing a file
try :
   file.open(file_path, 'mode') # opens the file
   # perform file operations here
   content = file.read() # reads the entire content of the file
   print(content)
finally:
    file.close() # ensures the file is closed even if an error occurs  
    print("File closed in finally block.")


# file types : CSV, JSON, XML, Binary
file = open('data.csv', 'r') # opens the CSV file in read mode
csv_content = file.read() # reads the entire content of the CSV file
print(csv_content)
file.close() # closes the file

file = open('data.json', 'r') # opens the JSON file in read mode
json_content = file.read() # reads the entire content of the JSON file 
print(json_content)
file.close() # closes the file  

file = open('data.xml', 'r') # opens the XML file in read mode
xml_content = file.read() # reads the entire content of the XML file
print(xml_content)
file.close() # closes the file

file = open('data.bin', 'rb') # opens the binary file in read-binary mode
binary_content = file.read() # reads the entire content of the binary file
print(binary_content)
file.close() # closes the file

