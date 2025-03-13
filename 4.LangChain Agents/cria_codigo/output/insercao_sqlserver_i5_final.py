
import pyodbc
import csv

SERVER = 'your_server_name'
DATABASE = 'your_database_name'
USERNAME = 'your_username'
PASSWORD = 'your_password'
TABLE_NAME = 'your_table_name'
COLUMN1 = 'column1'
COLUMN2 = 'column2'

cnxn_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}"

try:
    cnxn = pyodbc.connect(cnxn_string)
except pyodbc.Error as e:
    print(f"Error connecting to the database: {e}")
else:
    cursor = cnxn.cursor()

    with open('clips_count.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        query = f"INSERT INTO [{TABLE_NAME}] ([{COLUMN1}], [{COLUMN2}]) VALUES (?, ?)"
        data = [(row[0], row[1]) for row in reader]
        cursor.executemany(query, data)

    cnxn.commit()
    cnxn.close()
