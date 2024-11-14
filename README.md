# Driver Drowsiness Detection System
[https://www.linkedin.com/posts/jayasai2993_python-tensorflow-keras-activity-7228740669102563328-qx4-?utm_source=share&utm_medium=member_desktop]<br/><br/>

🚗 Driver Drowsiness Detection System<br/>
This project is a real-time driver drowsiness detection system using deep learning, computer vision, and Flask for web-based interaction. <br/> The system detects driver drowsiness by analyzing facial features and eye states from a video feed and raises an alert when drowsiness is detected.<br/><br/>

📑 Project Overview<br/>
This application employs a pre-trained deep learning model to detect drowsiness based on eye states in real-time. <br/> The system uses a Flask web server for user interaction, allowing the user to:<br/><br/>

📸 Upload an Image - Classifies the eye state in a single image as Drowsy or Not Drowsy. <br/>
🎥 Live Camera Feed - Monitors eye state in real-time through a webcam feed. <br/>     🔔 If drowsiness is detected for a prolonged period, an alert message and beep sound are triggered to warn the driver.<br/><br/>
🛠 Tech Stack<br/>
  Python 🐍 for backend logic and deep learning <br/>
  Flask 🌐 for the web server to host the application <br/>
  OpenCV 👁 for face and eye detection using Haar cascades <br/>
  TensorFlow/Keras 🤖 for deep learning model handling <br/>
  HTML, CSS, JavaScript 💻 for the front-end interface <br/><br/>
✨ Features<br/>
  🖼️ Image Upload for Drowsiness Classification <br/>     
   Users can upload an image, and the model will classify whether the person in the image is Drowsy or Not Drowsy. <br/>

🎥 Live Camera Monitoring <br/>     - Real-time monitoring via webcam to detect drowsiness, with an alert system that sounds an alarm and displays a warning on the video feed. <br/>
<br/>
📁 Project Structure
app.py 📜 - Main Flask application file containing routes and core logic. <br/>
driver_drowsiness_detection.h5 🧠 - Pre-trained Keras model for eye state classification. <br/>
static/uploads 📂 - Directory to store uploaded images. <br/>
templates/index.html 🌐 - Main HTML template for the web interface. <br/>
