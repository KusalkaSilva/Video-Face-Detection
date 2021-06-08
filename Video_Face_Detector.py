import numpy as np
import cv2
import sys

#Load the cascade
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#To capture video from webcam
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #Default webcam

#To use a video file as input
#webcam = cv2.VideoCapture('videoname.mp4')

if not webcam.isOpened():
    print("Cannot open camera")
    exit()

#Iterate forever over frames
while True:

    #Read the current frame
    ret, frame = webcam.read()
    frame=cv2.flip(frame,1,1) #Flip to act as a mirror

    if not ret:
        print("Can't recieve frame (stream end?). Exiting ...")
        break

    #Must convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    faces = trained_face_data.detectMultiScale(gray)

    #Draw rectangles around the faces
    for (x,y,w,h) in faces:
       cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
    #Display the image with the faces spotted
    cv2.imshow('Face Detector', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#Release the VideoCapture object
webcam.release()
cv2.destroyAllWindows()



