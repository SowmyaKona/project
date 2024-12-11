from flask import Flask, render_template, request, redirect, url_for
import cv2
import face_recognition
import sqlite3
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Function to load known faces and names
def load_known_faces():
    known_faces = []
    known_names = []
    for image_name in os.listdir('known_faces'):
        image_path = os.path.join('known_faces', image_name)
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(rgb_image)
        
        if face_encoding:
            known_faces.append(face_encoding[0])
            known_names.append(image_name.split('.')[0])  # Use filename as name
    return known_faces, known_names

# Function to connect to the SQLite database
def connect_db():
    conn = sqlite3.connect('database/attendance.db')
    return conn

# Function to mark attendance
def mark_attendance(name):
    conn = connect_db()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, timestamp))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_attendance')
def start_attendance():
    known_faces, known_names = load_known_faces()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = frame[:, :, ::-1]  # Convert to RGB
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
                mark_attendance(name)

            # Draw a rectangle around the face and label it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Attendance System", frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
