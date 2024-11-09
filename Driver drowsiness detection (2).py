#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
tf.random.set_seed(3)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from collections import Counter


# In[2]:


get_ipython().system('pip install split-folders')


# In[3]:


import splitfolders

data_dir = r'C:\Users\SAI\Downloads\archive (4)\Driver Drowsiness Dataset (DDD)'
output_dir = '/kaggle/working/splitted_Data'
splitfolders.ratio(data_dir, output=output_dir, seed=1337, ratio=(.8, 0.15, 0.05))


# In[4]:


train_dir = "/kaggle/working/splitted_Data/train"
test_dir = "/kaggle/working/splitted_Data/test"
val_dir = "/kaggle/working/splitted_Data/val"


# In[5]:


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)


# In[6]:


train_batches = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=True
)


# In[7]:


test_batches = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary' ,
    shuffle=True
)


# In[8]:


val_batches = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=True
)


# In[9]:


train_class_indices = train_batches.class_indices
test_class_indices = test_batches.class_indices
val_class_indices = val_batches.class_indices

train_class_labels = train_batches.classes
test_class_labels = test_batches.classes
val_class_labels = val_batches.classes


train_class_counts = Counter(train_class_labels)
test_class_counts = Counter(test_class_labels)
val_class_counts = Counter(val_class_labels)

print("Class Names for train:\n", train_class_indices)
print("Class Counts for train:\n", train_class_counts)
print(end='\n')

print("Class Names for test:\n", test_class_indices)
print("Class Counts for test:\n", test_class_counts)
print(end='\n')

print("Class Names for validation :\n", val_class_indices)
print("Class Counts for validation:\n", val_class_counts)


# In[10]:


images, labels = next(train_batches)
print(f"Pixels of the first image after Normalization: \n\n{images[0]}") #print pixels of the first img
plt.imshow(images[0])
plt.show()


# In[11]:


print(f"there are { images[0].ndim} Channels ")
print(f"image shape : {images[0].shape}")


# In[12]:


fig, axes = plt.subplots(8, 4, figsize=(15, 30))
class_indices = train_batches.class_indices

for i in range(8):
    images, labels = next(train_batches)
    for j in range(4):
        
        ax = axes[i, j]
        ax.imshow(images[j])
        ax.axis('off')
        label = int(labels[j])  
        label_name = list(class_indices.keys())[list(class_indices.values()).index(label)]
        ax.set_title(f'{label_name} ({label})')

plt.tight_layout()
plt.show()


# In[14]:


from tensorflow.keras.applications import MobileNetV2

# Define the image size
image_size = (224, 224)  # Example size, adjust according to your dataset

# Create the base model
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(image_size[0], image_size[1], 3),
)


# In[15]:


type(base_model)


# In[16]:


base_model.summary()


# In[17]:


model=keras.Sequential() #empty


# In[18]:


for layer in base_model.layers[:-25] :
    layer.trainable = False


# In[19]:


x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)


# In[20]:


model = Model(inputs=base_model.input, outputs=predictions)


# In[21]:


model.summary()


# In[23]:


model.compile(optimizer=Adam(0.0001 ), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[24]:


history = model.fit(
   train_batches,
    epochs=5,
    validation_data=val_batches,
    batch_size=32
)


# In[25]:


tf.keras.utils.plot_model(model, show_shapes=True)


# In[26]:


loss, accuracy = model.evaluate(train_batches)

print(f"Training Loss: {loss:.4f}")
print(f"Training Accuracy: {accuracy*100:.2f}%")


# In[27]:


loss, accuracy = model.evaluate(test_batches)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")


# In[28]:


y_pred = model.predict(test_batches)

y_pred_labels = np.argmax(y_pred, axis=1)


# In[29]:


y_actual = test_batches.labels


# In[30]:


conf_matrix = confusion_matrix(y_actual, y_pred_labels)

print(conf_matrix)


# In[31]:


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='bone', 
            xticklabels=['Class 0', 'Class 1'], 
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[32]:


report = classification_report(y_actual, y_pred_labels)
print(report)


# In[42]:


model.save('driver_drowsiness_detection.h5')


# In[44]:


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Load the model
model = load_model('driver_drowsiness_detection.h5')

# Class indices mapping
class_indices = {0: 'Drowsy', 1: 'Not Drowsy'}  # Update with your actual class names

# Preprocess image function
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale
    return img_array

# Preprocess frame function
def preprocess_frame(frame, target_size=(224, 224)):
    frame = cv2.resize(frame, target_size)
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Predict on new image
img_path = r"C:\Users\SAI\Desktop\test1.webp"
preprocessed_image = preprocess_image(img_path)
prediction = model.predict(preprocessed_image)
predicted_class = np.argmax(prediction, axis=1)
predicted_class_name = class_indices[predicted_class[0]]
print(f"Predicted class: {predicted_class_name}")
# Predict on video with batch processing
def predict_on_video(video_path, model, batch_size=32):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        preprocessed_frame = preprocess_frame(frame)
        frames.append(preprocessed_frame)

        # Process batch
        if len(frames) == batch_size:
            batch_predictions = model.predict(np.array(frames))
            predicted_classes = np.argmax(batch_predictions, axis=1)
            predictions.extend(predicted_classes)
            frames = []  # Reset frames list

        # Display the frame with prediction
        class_name = class_indices[predicted_classes[-1]]
        cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Process any remaining frames
    if frames:
        batch_predictions = model.predict(np.array(frames))
        predicted_classes = np.argmax(batch_predictions, axis=1)
        predictions.extend(predicted_classes)

    cap.release()
    cv2.destroyAllWindows()
    return predictions

# Example of predicting on a video
video_path = 'path_to_your_video.mp4'
predictions = predict_on_video(video_path, model)
print(predictions)


# In[46]:


import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
start_time = time.time()
model = load_model('driver_drowsiness_detection.h5')
print(f"Model loading time: {time.time() - start_time:.2f} seconds")

# Class indices mapping
class_indices = {0: 'drowsy', 1: 'not drowsy'}  # Update with your actual class names

# Preprocess image function
def preprocess_image(img_path, target_size=(224, 224)):
    start_time = time.time()
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale
    print(f"Image preprocessing time: {time.time() - start_time:.2f} seconds")
    return img_array

# Example of loading and preprocessing a new image
img_path = r'C:\Users\SAI\Desktop\test1.webp'
preprocessed_image = preprocess_image(img_path)

# Predict the class of the new image
start_time = time.time()
prediction = model.predict(preprocessed_image)
print(f"Prediction time: {time.time() - start_time:.2f} seconds")

predicted_class = np.argmax(prediction, axis=1)
predicted_class_name = class_indices[predicted_class[0]]
print(f"Predicted class: {predicted_class_name}")


# In[1]:


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time
from tensorflow.keras.preprocessing import image as keras_image

# Load the pre-trained model for drowsiness detection
start_time = time.time()
model = load_model('driver_drowsiness_detection.h5')
print(f"Model loading time: {time.time() - start_time:.2f} seconds")

# Class indices mapping
class_indices = {0: 'drowsy', 1: 'not drowsy'}  # Update with your actual class names

# Preprocess image function
def preprocess_image(img_path, target_size=(224, 224)):
    start_time = time.time()
    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale
    print(f"Image preprocessing time: {time.time() - start_time:.2f} seconds")
    return img_array

# Load the Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_face_and_eyes(image):
    # Convert the image to grayscale for face and eye detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Region of interest for eyes detection within the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # Detect eyes within the face
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Filter out false positives by size and position within the face
        for (ex, ey, ew, eh) in eyes:
            # Condition to filter out false positives (e.g., nose)
            if ey < h / 2:  # Only consider eyes in the upper half of the face
                # Draw rectangle around each eye
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    return image

# Function to display the image with face and eye detections
def display_image_with_detections(img_path):
    image = cv2.imread(img_path)
    image_with_detections = detect_face_and_eyes(image)
    return image_with_detections

# Example of loading and preprocessing a new image
img_path = r"C:\Users\SAI\Desktop\t3.jpeg"
preprocessed_image = preprocess_image(img_path)

# Predict the class of the new image
start_time = time.time()
prediction = model.predict(preprocessed_image)
print(f"Prediction time: {time.time() - start_time:.2f} seconds")

predicted_class = np.argmax(prediction, axis=1)
predicted_class_name = class_indices[predicted_class[0]]
print(f"Predicted class: {predicted_class_name}")

# Display the image with detections
result_image = display_image_with_detections(img_path)

# Convert the result image from BGR to RGB for matplotlib display
result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

# Display the result image using matplotlib
plt.imshow(result_image_rgb)
plt.axis('off')  # Hide axes ticks
plt.title(f'Prediction: {predicted_class_name}')
plt.show()


# In[1]:


get_ipython().system('pip uninstall opencv-python -y')


# In[2]:


get_ipython().system('pip uninstall opencv-python-headless -y')


# In[3]:


get_ipython().system('pip install opencv-python')


# In[7]:


import cv2
import numpy as np
from collections import deque

# Load the Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_pupil(eye_region):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    
    # Adaptive thresholding for better pupil detection
    thresh_eye = cv2.adaptiveThreshold(blurred_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours to detect pupils
    contours, _ = cv2.findContours(thresh_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if 50 < cv2.contourArea(largest_contour) < 500:  # Filter out very small and very large contours
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(eye_region, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.drawContours(eye_region, [largest_contour], -1, (0, 255, 0), 2)
            return True
    return False

def detect_face_and_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_closed = False

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Region of interest for eyes detection within the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # Detect eyes within the face
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        eyes_closed_in_frame = True
        
        for (ex, ey, ew, eh) in eyes:
            if ey < h / 2:  # Only consider eyes in the upper half of the face
                # Draw rectangle around each eye
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                # Extract the eye region for further processing
                eye_region = roi_color[ey:ey+eh, ex:ex+ew]
                
                # Detect pupil
                pupil_detected = detect_pupil(eye_region)
                
                # Check if pupil is detected
                if pupil_detected:
                    eyes_closed_in_frame = False
                    cv2.putText(roi_color, 'open', (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        if eyes_closed_in_frame:
            eyes_closed = True

    return eyes_closed

# Start video capture from webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Buffer to track eye state over time
buffer_size = 25  # Buffer size set to 8 frames
eye_state_buffer = deque(maxlen=buffer_size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    eyes_closed = detect_face_and_eyes(frame)

    # Add current eye state to the buffer
    eye_state_buffer.append(eyes_closed)

    # Debugging: Print buffer contents and current eye state
    print(f"Current Eye State: {'Closed' if eyes_closed else 'Open'}")
    print(f"Eye State Buffer: {list(eye_state_buffer)}")

    # Determine overall drowsiness state based on buffer
    if len(eye_state_buffer) == buffer_size and all(eye_state_buffer):
        final_prediction = 'Drowsy'
    else:
        final_prediction = 'Not Drowsy'
    
    # Overlay the prediction label on the image
    cv2.putText(frame, f'Prediction: {final_prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Drowsiness Detection', frame)

    # Write the frame to the output video file
    out.write(frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Live camera feed ended and output video saved.")


# In[ ]:




