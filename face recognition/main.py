import cv2
import numpy as np
import face_recognition as face_rec
import os
import pyttsx3 as textSpeech
from datetime import datetime

engine = textSpeech.init()

def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

path = 'images'
studentImg = []
studentName = []
myList = os.listdir(path)
for cl in myList:
    curimg = cv2.imread(f'{path}/{cl}')
    studentImg.append(curimg)
    studentName.append(os.path.splitext(cl)[0])

def findEncoding(images):
    imgEncodings = []
    for img in images:
        img = resize(img, 1.0)  # Increased resize factor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img, model="cnn")[0]  # Use CNN model
        imgEncodings.append(encodeimg)
    return imgEncodings

def MarkAttendance(name):
    now = datetime.now()
    timestr = now.strftime('%H:%M')
    with open('attendance.csv', 'a') as f:  # Use 'a' mode to append
        f.write(f'\n{name}, {timestr}')
    statement = f'Welcome to class, {name}'
    engine.say(statement)
    engine.runAndWait()

EncodeList = findEncoding(studentImg)

vid = cv2.VideoCapture(1)
while True:
    success, frame = vid.read()
    Smaller_frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

    facesInFrame = face_rec.face_locations(Smaller_frames)
    encodeFacesInFrame = face_rec.face_encodings(Smaller_frames, facesInFrame, model="cnn")  # Use CNN model

    for encodeFace, faceLoc in zip(encodeFacesInFrame, facesInFrame):
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        faceDis = face_rec.face_distance(EncodeList, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.5:  # Adjusted threshold
            name = studentName[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            MarkAttendance(name)

    cv2.imshow('video', frame)
    cv2.waitKey(1)
