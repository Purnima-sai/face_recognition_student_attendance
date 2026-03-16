# Face Recognition Attendance System

A Django-based web application that automates attendance tracking using facial recognition technology.  
The system captures user images, trains a machine learning model using facial embeddings, and identifies individuals in real time through a webcam stream to automatically record attendance.

---

## Tech Stack

### Backend
- **Django (Python)** – Handles application logic, routing, database models, and the admin interface.
- **SQLite** – Lightweight relational database used for storing student/employee information and attendance records.

### Computer Vision
- **OpenCV** – Used for webcam access, frame capture, image preprocessing, and real-time video stream processing.
- **face_recognition (dlib-based)** – Detects faces and generates facial encodings for recognition.

### Machine Learning
- **scikit-learn (K-Nearest Neighbors – KNN)** – Used to train a classifier on facial embeddings to identify individuals.

### Frontend
- **HTML5** – Page structure.
- **CSS3** – Styling and layout.
- **Bootstrap** – Responsive UI design.
- **JavaScript & jQuery** – Interactive UI behavior and webcam integration.

### Supporting Libraries
- **NumPy** – Numerical processing of image arrays.
- **Pillow** – Image handling and preprocessing.
- **Joblib** – Saving and loading trained machine learning models.

---

## System Workflow

### 1. User Registration
The administrator registers students or employees through the Django interface by entering basic details such as name and ID.

### 2. Photo Capture
The system captures multiple facial images for each registered individual using a webcam.  
These images are stored in a structured dataset where each person has a dedicated folder.

### 3. Dataset Preparation
Captured images are processed using the `face_recognition` library to extract **facial encodings**, which represent unique numerical features of each face.

### 4. Model Training
A **K-Nearest Neighbors (KNN)** classifier is trained using the extracted facial encodings.  
The model learns the relationship between facial feature vectors and their corresponding identities.

Multiple model configurations can be trained to balance **recognition speed and accuracy**.

### 5. Real-Time Face Detection
Using OpenCV, the webcam stream is processed frame by frame to detect faces in the video feed.

### 6. Face Encoding
Detected faces are converted into facial embeddings using the `face_recognition` library.

### 7. Face Identification
The trained KNN model compares detected embeddings with known encodings in the dataset and predicts the closest matching identity.

### 8. Attendance Logging
When a face is successfully recognized, the system automatically records:
- Person Name / ID
- Detection Timestamp
- Captured Face Image

This information is stored in the database as an attendance record.

### 9. Data Management
Administrators can view and manage attendance logs through the Django dashboard and export records for reporting or analysis.

---

## Machine Learning Pipeline

The recognition pipeline follows these stages:

1. **Face Detection** – Detect faces in each frame.
2. **Face Encoding** – Convert faces into numerical feature vectors.
3. **Model Prediction** – Classify faces using the trained KNN model.
4. **Confidence Evaluation** – Verify the prediction accuracy.
5. **Attendance Recording** – Store recognized identity with timestamp.

---

## Performance Considerations

- Frame skipping is used during video processing to reduce computational load.
- Using multiple training images per individual improves recognition accuracy.
- Higher jitter values during training improve model robustness but increase training time.
- Retraining the model periodically helps maintain accuracy when new users are added.

---

## Core Functional Modules

**Photo Capture Module**  
Captures and stores multiple images for each user.

**Training Module**  
Processes the dataset and trains the facial recognition model.

**Recognition Module**  
Handles real-time face detection and identity prediction.

**Attendance Module**  
Records recognized individuals with timestamps and stores them in the database.
