# face_recognition_student_attendance

A Django-based web application that automates attendance tracking using facial recognition. The system captures user images, trains a machine learning model using facial embeddings, and identifies individuals in real time through a webcam stream to automatically record attendance.

Tech Stack
Backend

Django (Python) – Handles the application logic, routing, database models, and admin management.

SQLite – Lightweight relational database used for storing employee/student records and attendance logs.

Computer Vision

OpenCV – Used for webcam access, frame capture, image preprocessing, and real-time video stream handling.

face_recognition (dlib-based) – Extracts facial embeddings and performs face detection and encoding.

Machine Learning

scikit-learn (K-Nearest Neighbors – KNN) – Used to train a classifier on facial embeddings for recognizing individuals.

Frontend

HTML5 & CSS3 – Structure and styling of the user interface.

Bootstrap – Responsive layout and UI components.

JavaScript & jQuery – Handles dynamic UI interactions and webcam operations.

Supporting Libraries

NumPy – Numerical operations on image arrays.

Pillow – Image processing support.

Joblib – Saving and loading trained machine learning models.

System Workflow
1. User Registration

The administrator registers employees or students through the Django interface by entering details such as name and ID.

2. Photo Capture

The system captures multiple images of each registered person using the webcam. These images are stored in a structured dataset directory where each individual has a dedicated folder.

3. Dataset Creation

Captured images are processed using the face_recognition library to extract facial encodings. These encodings represent unique facial features and form the dataset used for training.

4. Model Training

A K-Nearest Neighbors (KNN) classifier is trained using the extracted facial encodings.
The model learns to associate facial feature vectors with corresponding identities.

Multiple model configurations can be trained to balance between recognition speed and accuracy.

5. Real-Time Face Detection

The webcam stream is processed frame by frame using OpenCV. Faces detected in each frame are converted into embeddings using the face recognition library.

6. Face Identification

The trained KNN model compares detected facial embeddings with known encodings in the dataset and predicts the most likely identity.

7. Attendance Logging

Once a face is recognized, the system automatically records:

Person name/ID

Detection timestamp

Captured face image

These records are stored in the database for tracking attendance.

8. Data Management

Administrators can review attendance logs, detected faces, and user records through the Django dashboard. Attendance data can also be exported for reporting and analysis.

Machine Learning Pipeline

The recognition pipeline follows these stages:

Face Detection – Locate faces in the image frame.

Face Encoding – Convert detected faces into numerical embeddings.

Model Prediction – Use KNN to classify the face.

Confidence Evaluation – Verify prediction accuracy.

Attendance Recording – Store recognized identity with timestamp.

Performance Considerations

Frame skipping is used during video processing to reduce computational load.

Multiple training images per individual improve recognition accuracy.

Higher jitter values during training improve model robustness but increase processing time.

Periodic retraining ensures the model adapts when new users are added.

Core Functional Modules

Photo Capture Module
Captures and stores multiple facial images for each user.

Training Module
Processes the dataset and trains the KNN recognition model.

Recognition Module
Handles real-time face detection and identity prediction.

Attendance Module
Records recognized individuals with timestamps and stores the data in the database.
