from flask import Flask, render_template, request, redirect
import cv2
import os
import face_recognition
from datetime import datetime
import mysql.connector
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





app = Flask(__name__)

CAPTURE_FOLDER = 'static/captures'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        student_name = request.form['student_name'].strip()
        if student_name:
            save_faces_from_camera(student_name)
            return redirect('/')
        
    return render_template('index.html')

@app.route('/recognize')
def recognize():
    known_faces = []
    known_names = []

    # Load known faces
    for name in os.listdir(CAPTURE_FOLDER):
        for filename in os.listdir(os.path.join(CAPTURE_FOLDER, name)):
            image_path = os.path.join(CAPTURE_FOLDER, name, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_faces.append(encodings[0])
                known_names.append(name)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]
        locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
            match = None

            if True in results:
                match = known_names[results.index(True)]
                mark_attendance(match)
                match = known_names[results.index(True)]
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, match, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Recognizing Faces - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect('/attendance')
@app.route('/attendance')
def attendance():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT s.name, a.timestamp 
        FROM students s
        JOIN attendance a ON s.id = a.student_id
        ORDER BY a.timestamp DESC
    """)
    records = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('attendance.html', records=records)


def save_faces_from_camera(student_name):
    folder_path = os.path.join(CAPTURE_FOLDER, student_name)
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while count < 5:  # Capture 5 face images
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)

        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_image = frame[top:bottom, left:right]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{student_name}_{timestamp}_{i}.jpg"
            cv2.imwrite(os.path.join(folder_path, filename), face_image)
            count += 1

        cv2.imshow("Capturing Faces - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True)
