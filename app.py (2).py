#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import threading
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from collections import deque
from flask_ngrok import run_with_ngrok
import winsound  # Import this for the beep sound on Windows

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run

# Load the pre-trained model for image classification
model_path = r"C:\Users\SAI\driver_drowsiness_detection.h5"
model = load_model(model_path)

# Class indices mapping
class_indices = {0: 'drowsy', 1: 'not drowsy'}

# Function to preprocess image
def preprocess_image(img_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale
    return img_array

# Load the Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to detect pupils
def detect_pupil(eye_region):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    thresh_eye = cv2.adaptiveThreshold(blurred_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if 50 < cv2.contourArea(largest_contour) < 500:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(eye_region, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.drawContours(eye_region, [largest_contour], -1, (0, 255, 0), 2)
            return True
    return False

# Function to detect face and eyes
def detect_face_and_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes_closed = False
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        eyes_closed_in_frame = True
        for (ex, ey, ew, eh) in eyes:
            if ey < h / 2:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                eye_region = roi_color[ey:ey+eh, ex:ex+ew]
                pupil_detected = detect_pupil(eye_region)
                if pupil_detected:
                    eyes_closed_in_frame = False
                    cv2.putText(roi_color, 'open', (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if eyes_closed_in_frame:
            eyes_closed = True
    return eyes_closed

# Initialize the video capture
cap = None

# Buffer to track eye state over time
buffer_size = 20
eye_state_buffer = deque(maxlen=buffer_size)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        img_path = os.path.join('static', 'uploads', file.filename)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        file.save(img_path)
        preprocessed_image = preprocess_image(img_path)
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_class_name = class_indices[predicted_class[0]]
        return render_template('index.html', prediction=predicted_class_name, image_path=img_path)

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    return jsonify(status='Camera started')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    return jsonify(status='Camera stopped')


def gen():
    global cap
    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
            eyes_closed = detect_face_and_eyes(frame)
            eye_state_buffer.append(eyes_closed)
            if len(eye_state_buffer) == buffer_size and all(eye_state_buffer):
                final_prediction = 'Drowsy'
                # Draw the "Alert!" text at the center of the frame
                height, width, _ = frame.shape
                cv2.putText(frame, 'ALERT!', (width // 2 - 100, height // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
                # Play a beep sound
                winsound.Beep(1000, 500)  # Frequency = 1000 Hz, Duration = 500 ms
            else:
                final_prediction = 'Not Drowsy'
            # Display the prediction
            cv2.putText(frame, f'Prediction: {final_prediction}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run()
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


# In[ ]:




