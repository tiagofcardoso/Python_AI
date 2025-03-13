
import pyodbc
import csv

# Define connection parameters
server = 'your_server_name'
database = 'your_database_name'
username = 'your_username'
password = 'your_password'
table_name = 'your_table_name'
column1 = 'column1'
column2 = 'column2'

# Establish the connection
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)

# Create a cursor object
cursor = cnxn.cursor()

# Open the CSV file
with open('clips_count.csv', 'r') as f:
    reader = csv.reader(f)
    
    # Iterate over each row in the CSV file
    for row in reader:
        query = "INSERT INTO [{}] ([{}], [{}]) VALUES (?, ?)".format(table_name, column1, column2)
        cursor.execute(query, (row[0], row[1]))

# Commit the changes
cnxn.commit()

# Close the connection
cnxn.close()
