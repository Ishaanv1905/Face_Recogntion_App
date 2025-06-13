#main code
from registration import register_user
from attendance import mark_attendance
import sqlite3

with sqlite3.connect('App_Database.db') as conn:
    cursor = conn.cursor()
    cursor.execute(''' CREATE TABLE IF NOT EXISTS employees (
    employee_id TEXT PRIMARY KEY,
    name TEXT,
    embedding TEXT)''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT,
        name TEXT,
        timestamp TEXT)''')
    conn.commit()

while True:
    print("\n--- Face Recognition Attendance System ---")
    print("1. New User Registration")
    print("2. Give Attendance")
    print("3. Exit")
    choice = input("Enter your choice: ")
    if choice == '1':
        register_user()
    elif choice == '2':
        mark_attendance()
    elif choice == '3':
        print("Exiting...")
        break
    else:
        print("Invalid choice. Try again.")
