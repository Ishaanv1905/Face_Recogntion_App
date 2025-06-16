# registration code
from face_recognition import capture_face
import numpy as np
import sqlite3

def register_user():
    employee_id = input("Enter Employee ID: ")
    name = input("Enter Employee Name: ")
    print("Please look at the camera. Take 3 photos.")      
    embeddings = capture_face()
    if embeddings is None:
        return
    avg_embedding = np.mean(embeddings, axis=0)    
    save_employee(employee_id, name, avg_embedding)
        
def save_employee(emp_id, name, embedding):
    embedding_str = ' '.join(map(str, np.array(embedding).flatten()))    
    with sqlite3.connect('App_Database.db') as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO employees (employee_id, name, embedding) VALUES (?, ?, ?)", (emp_id, name, embedding_str))
        conn.commit()
    print(f"Employee {name} saved to database.")
