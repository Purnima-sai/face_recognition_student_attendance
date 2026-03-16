from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils import timezone
import pytz
from .facerec.faster_video_stream import stream
from .facerec.click_photos import click
from .facerec.train_faces import trainer
from .models import Employee, Detected
from .forms import EmployeeForm
import cv2
import pickle
import face_recognition
import datetime
from cachetools import TTLCache
import numpy
import os
import csv
import sqlite3
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


cache = TTLCache(maxsize=20, ttl=60)


def identify1(frame, name, buf, buf_length, known_conf):

    if name in cache:
        return
    count = 0
    for ele in buf:
        count += ele.count(name)
    
    if count >= known_conf:
        timestamp = datetime.datetime.now(tz=pytz.UTC)
        print(name, timestamp)
        cache[name] = 'detected'
        path = 'detected/{}_{}.jpg'.format(name, timestamp)
        write_path = 'media/' + path
        cv2.imwrite(write_path, frame)
        try:
            emp = Employee.objects.get(name=name)
            emp.detected_set.create(time_stamp=timestamp, photo=path)
        except:
            pass 	        




def predict(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.5):

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    # X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # print(closest_distances)
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]



def identify_faces(video_capture):

    buf_length = 10
    known_conf = 6
    buf = [[]] * buf_length
    i = 0

    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_frame = small_frame[:, :, ::-1]
        rgb_frame = numpy.ascontiguousarray(small_frame[:, :, ::-1])

        if process_this_frame:
            predictions = predict(rgb_frame, model_path="app/facerec/models/trained_model.clf")
            # print(predictions)

        process_this_frame = not process_this_frame

        face_names = []

        for name, (top, right, bottom, left) in predictions:

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            identify1(frame, name, buf, buf_length, known_conf)

            face_names.append(name)

        buf[i] = face_names
        i = (i + 1) % buf_length


        # print(buf)


        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()



def index(request):
    return render(request, 'app/index.html')


def video_stream(request):
    stream()
    return HttpResponseRedirect(reverse('index'))


def add_photos(request):
	emp_list = Employee.objects.all()
	return render(request, 'app/add_photos.html', {'emp_list': emp_list})


def click_photos(request, emp_id):
	cam = cv2.VideoCapture(0)
	emp = get_object_or_404(Employee, id=emp_id)
	click(emp.name, emp.id, cam)
	return HttpResponseRedirect(reverse('add_photos'))


def train_model(request):
	trainer()
	return HttpResponseRedirect(reverse('index'))


def detected(request):
	if request.method == 'GET':
		date_formatted = datetime.datetime.today().date()
		date = request.GET.get('search_box', None)
		if date is not None:
			date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
		det_list = Detected.objects.filter(time_stamp__date=date_formatted).order_by('time_stamp').reverse()

	# det_list = Detected.objects.all().order_by('time_stamp').reverse()
	return render(request, 'app/detected.html', {'det_list': det_list, 'date': date_formatted})


def identify(request):
	video_capture = cv2.VideoCapture(0)
	identify_faces(video_capture)
	return HttpResponseRedirect(reverse('index'))


def add_emp(request):
    if request.method == "POST":
        form = EmployeeForm(request.POST)
        if form.is_valid():
            emp = form.save()
            # post.author = request.user
            # post.published_date = timezone.now()
            # post.save()
            return HttpResponseRedirect(reverse('index'))
    else:
        form = EmployeeForm()
    return render(request, 'app/add_emp.html', {'form': form})


def send_emaill(sender_email, sender_password, recipient_email, subject, body, attachment_path):
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject

    # Add body to email
    message.attach(MIMEText(body, 'html'))

    # Open the file to be sent
    with open(attachment_path, 'rb') as attachment:
        # Add file as application/octet-stream
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        'Content-Disposition',
        f'attachment; filename= {attachment_path.split("/")[-1]}'
    )

    # Add attachment to message
    message.attach(part)

    # Log in to SMTP server and send email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())


def SendMail(request):
    if request.method == "GET":
        conn = sqlite3.connect('db.sqlite3') 
        query = "SELECT * FROM app_detected"
        df = pd.read_sql_query(query, conn)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        profile = os.path.join(BASE_DIR, 'media') 
        csv_file_path = os.path.join(profile, 'data.csv')
        df.to_csv(csv_file_path)
        sender_email = 'sudhatestmail@gmail.com'
        sender_password = 'nhbjssyxvxbendzz'
        recipient_email = 'doodamnikhil@gmail.com'
        subject = "Attendance report from KLM-CSE"
        body = """
            <html>
            <body>
            <p>Auto Generated  <b> Attendance File</b> email sent from KLM-CSE.</p>
            </body>
            </html>
                """
        attachment_path = csv_file_path
        send_emaill(sender_email, sender_password, recipient_email, subject, body, attachment_path)


        return HttpResponseRedirect(reverse('index'))
    else:
        form = EmployeeForm()
    return render(request, 'app/add_emp.html', {'form': form})





def daySendMail(request):
    if request.method == "GET":
        date_formatted = datetime.datetime.today().date()
        date = request.GET.get('search_box', None)
        if date is not None:
            date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        det_list = Detected.objects.filter(time_stamp__date=date_formatted).order_by('time_stamp').reverse()
        
        df = pd.DataFrame(det_list, columns=['time_stamp', 'emp_id', 'photo'])
        # conn = sqlite3.connect('db.sqlite3') 
        # query = "SELECT * FROM app_detected where "
        # df = pd.read_sql_query(query, conn)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        profile = os.path.join(BASE_DIR, 'media') 
        csv_file_path = os.path.join(profile, 'data.csv')
        df.to_csv(csv_file_path)
        sender_email = 'sudhatestmail@gmail.com'
        sender_password = 'nhbjssyxvxbendzz'
        recipient_email = 'doodamnikhil@gmail.com'
        subject = "Attendance report from KLM-CSE"
        body = """
            <html>
            <body>
            <p>Auto Generated  <b> from the </b> Attendance management system using face recognition</p>
            </body>
            </html>
                """
        attachment_path = csv_file_path
        send_emaill(sender_email, sender_password, recipient_email, subject, body, attachment_path)


        return HttpResponseRedirect(reverse('index'))
    else:
        form = EmployeeForm()
    return render(request, 'app/add_emp.html', {'form': form})



