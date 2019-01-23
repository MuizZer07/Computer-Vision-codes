# -*- coding: utf-8 -*-

import cv2

# loading cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# detection
def detect(gray, frame):
    # returns tuples, x,y cordinates, width and height of the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  
    
    # for each faces, drawing a rectangle around 
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]  # zone within face, gray scale image
        roi_color = frame[y:y+h, x:x+w]  # color image
        
        # drawing rectangles around eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        
    return frame

# capturing photo from webcam and call the funcition
video_capture = cv2.VideoCapture(0)  # 0 for internal webcam, 1 for external webcam

while True:
    _ , frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_frame = detect(gray, frame)
    
    cv2.imshow('Video', new_frame)
    
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    
cv2.destroyAllWindows()
video_capture.release()