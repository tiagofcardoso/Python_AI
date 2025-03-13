
import pyodbc
import csv

SERVER = 'your_server_name'
DATABASE = 'your_database_name'
USERNAME = 'your_username'
PASSWORD = 'your_password'
TABLE_NAME = 'your_table_name'
COLUMN1 = 'column1'
COLUMN2 = 'column2'

cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + SERVER + ';DATABASE=' + DATABASE + ';UID=' + USERNAME + ';PWD=' + PASSWORD)

cursor = cnxn.cursor()

with open('clips_count.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header
    query = "INSERT INTO [{}] ([{}], [{}]) VALUES (?, ?)".format(TABLE_NAME, COLUMN1, COLUMN2)
    cursor.executemany(query, ((row[0], row[1]) for row in reader))

cnxn.commit()

cnxn.close()
