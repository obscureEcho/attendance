import face_recognition
import cv2
import os
import mysql.connector
from datetime import datetime

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="face_attendance"
    )

def mark_attendance(student_name):
    conn = get_db_connection()
    cursor = conn.cursor()

    student_name = student_name.strip().lower()

    # Insert student if not exists
    cursor.execute("INSERT IGNORE INTO students (name) VALUES (%s)", (student_name,))
    conn.commit()

    # Get student ID
    cursor.execute("SELECT id FROM students WHERE name = %s", (student_name,))
    result = cursor.fetchone()

    if result:
        student_id = result[0]
        today = datetime.now().date()

        # Check if already marked today
        cursor.execute("""
            SELECT COUNT(*) FROM attendance 
            WHERE student_id = %s AND DATE(timestamp) = %s
        """, (student_id, today))

        already_marked = cursor.fetchone()[0]

        if not already_marked:
            cursor.execute("INSERT INTO attendance (student_id) VALUES (%s)", (student_id,))
            conn.commit()
            print(f"✅ Attendance marked for {student_name}")
        else:
            print(f"🟡 Already marked today for {student_name}")
    else:
        print(f"❌ Student not found: {student_name}")

    cursor.close()
    conn.close()



KNOWN_FACES_DIR = "static/captures"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"  # or "cnn" for GPU

print("Loading known faces...")

known_faces = []
known_names = []

# Load known faces
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_faces.append(encodings[0])
            known_names.append(name)

print("Starting webcam...")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]
    locations = face_recognition.face_locations(rgb_frame, model=MODEL)
    encodings = face_recognition.face_encodings(rgb_frame, locations)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None

        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
            mark_attendance(match)
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS)
            cv2.putText(frame, match, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), FONT_THICKNESS)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
