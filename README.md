# Driver Drowsiness Detection System
[https://www.linkedin.com/posts/jayasai2993_python-tensorflow-keras-activity-7228740669102563328-qx4-?utm_source=share&utm_medium=member_desktop]<br/><br/>


- This project is a real-time driver drowsiness detection system using deep learning, computer vision, and Flask for web-based interaction. The system can detect a driver's drowsiness by analyzing facial features and eye states from a video feed and raise an alert when drowsiness is detected.

- Project Overview
This application employs a pre-trained deep learning model to detect drowsiness based on eye states in real-time. The system uses a Flask web server for user interaction, allowing the user to:

- Upload an Image - Classifies the eye state in a single image as Drowsy or Not Drowsy.<br/>
- Live Camera Feed - Monitors eye state in real-time through a webcam feed. If drowsiness is detected for a prolonged period, an alert message and beep sound are triggered to warn the driver.<br/>
- Tech Stack<br/>
Python for backend logic and deep learning.<br/>
Flask for the web server to host the application.<br/>
OpenCV for face and eye detection, using Haar cascades.<br/>
TensorFlow/Keras for deep learning model handling.<br/>
HTML, CSS, JavaScript for the front-end.<br/><br/>
Features<br/>
Image Upload for Drowsiness Classification:<br/>
Users can upload an image, and the model will classify whether the person in the image is drowsy or not.<br/>
Live Webcam Monitoring:<br/>
A live video feed from the webcam detects the driver’s eye state.
Continuous drowsiness triggers an on-screen alert and a beep sound to warn the driver.<br/><br/>
Project Structure
app.py: Main Flask application file.
driver_drowsiness_detection.h5: Pre-trained model for drowsiness classification.
templates/: Contains index.html for the main interface.
static/uploads/: Directory for saving uploaded images.
Working Flow
1. Loading the Model and Dependencies
Load a pre-trained deep learning model for eye state classification.
Initialize OpenCV’s Haar cascades for detecting face and eye regions.
2. Web Interface
Homepage (index.html): Users can access options to upload an image for classification or initiate a live camera feed for real-time monitoring.
3. Image Upload
The user can upload an image for classification.
The image is preprocessed and fed to the model.
The classification result (Drowsy or Not Drowsy) is displayed along with the uploaded image on the interface.
4. Live Camera Feed for Real-Time Monitoring
Users can start the live camera feed to monitor the eye state continuously.
The application analyzes frames in real-time and detects if eyes are closed for an extended period.
If continuous closed eyes are detected, the system:
Displays an "ALERT!" message on the video feed.
Plays a beep sound to warn the driver of potential drowsiness.
5. Real-Time Face and Eye Detection
Uses Haar cascades to detect faces and eyes in each video frame.
Tracks eye state over time with a buffer of frames. If eyes are closed consistently across the buffer, the user is flagged as drowsy.
