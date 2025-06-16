#attendance code
from face_recognition import capture_face
import cv2
import numpy as np
from datetime import datetime
import sqlite3

def load_employees():
    with sqlite3.connect('App_Database.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT employee_id, name, embedding FROM employees")
        rows = cursor.fetchall()

    employees = []
    for row in rows:
        employee_id, name, embedding_str = row
        embedding = np.fromstring(embedding_str, sep=' ')
        employees.append({'employee_id': employee_id, 'name': name, 'embedding': embedding})

    return employees

def mark_attendance():
    employees = load_employees()
    if not employees:
        print("No Employees registered yet")
        return
    print("Camera starting... Press 'q' to exit.")
    cap = cv2.VideoCapture(0)
    threshold = 0.9  
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            cap.release()
            break

        captured_embedding = capture_face(frame_input=frame)

        if captured_embedding is not None:
            min_distance = float('inf')
            index = -1

            for i, emp in enumerate(employees):
                emp_embedding = emp['embedding']
                distance = np.linalg.norm(emp_embedding - captured_embedding)
                if distance < min_distance:
                    min_distance = distance
                    index = i

            cv2.putText(frame, f"Distance: {min_distance:.2f}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if min_distance < threshold:
                emp_id = employees[index]['employee_id']
                name = employees[index]['name']

                cv2.putText(frame, f"Welcome {name}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                log_attendance(emp_id, name)
                #print(distance)
                cv2.imshow('Attendance', frame)
                cv2.waitKey(3000)
                break
            else:
                cv2.putText(frame, "Face Not Recognized", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Attendance not marked. Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

def log_attendance(emp_id, name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect('App_Database.db') as conn:
        cursor = conn.cursor()
        cursor.execute(''' INSERT INTO attendance (employee_id, name, timestamp) VALUES (?, ?, ?)
        ''', (emp_id, name, timestamp))
        conn.commit()

    print(f"✅ Attendance marked for {name} at {timestamp}")

