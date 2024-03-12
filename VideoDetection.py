import os
import numpy as np
from ultralytics import YOLO
import cv2

#This creates the path to my video and outputs the video.
video_path = os.path.join('.', 'DemoVideo.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if frame is not None:
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

#This is the path to my model that I created. I collected my own data and trained the object dectector.
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

# Loads my custom hand model
model = YOLO(model_path)  

# Threshold for when the object is considered to be detected.
threshold = 0.5

class_name_dict = {0: 'hand'}

while ret:
    #******This part detects the hands using my custom model.*******
    #***************************************************************
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    #*******This part detects the head and shoulders*****    
    #****************************************************
    #Convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Loads the data for face detection.
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_defaultcopy.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #If there were no faces detected it can't detect shoulders so it has a condition to run.
    if len(faces) != 0:
        for(x, y, w, h) in faces:
            #The first tuple is the top-left corner of the box.
            #The second tuple is the bottom-right corner of the box.
            cv2.rectangle (frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            #I'm going to detect the shoulder by finding the shoulders relative to the 
            #rectangle of the head. This design heavily depends on the face being detected correctly.
            #Another flaw that I noticed was that the head had to be straight so it could detect the 
            #shoulders properly. The quality also decreased when there were multiple people on an image/frame
            #because it could be blocking the shoulder regions of one another.

            #****************************************************************
            #*****This sections out the left shoulder of the subject.********
            x1 = x + w
            y1 = y + h

            #img[height, width] to box out region of left shoulder
            left_shoulder = frame[y1 : y1 + h, x1 : x1 + w]
            left_gray = cv2.cvtColor(left_shoulder, cv2.COLOR_BGR2GRAY)

            #In order to find a line that keep track of the shoulders, I would first need to find the edge of the
            #shoulder with an edge detection technique. 
            left_edges = cv2.Canny(left_gray, 100, 200)
            
            #The way I thought of detecting the shoulders was traveling the shoulder regions
            #from left to right. I would then examine each column and check where there is a gradient change.
            #I would store points where there is a gradient change and then make a line of best fit to keep track 
            #of the shoulders. 
            left_points = []
            for x_row_l in range (left_edges.shape[1]):
                for y_column_l in range(left_edges.shape[0]):
                    if left_edges[y_column_l,x_row_l] > 0:
                        left_points.append((x_row_l, y_column_l))

            #Extracted the x and y values from the points to use np.polyfit().
            left_x = np.array([p[0] for p in left_points])  
            left_y = np.array([p[1] for p in left_points])

            #I used the slope to check if shrugging was present. The color will be blue if there's
            #no shrugging. It will change to red if there is shrugging.
            #It will also not run if there is no data to make the line of best fit.
            if len(left_x) != 0 and len(left_y) != 0:
                slope, intercept = np.polyfit (left_x, left_y , 1)
                if 0.3 <= slope <= 0.8:
                    cv2.line(left_shoulder, (0, int(intercept)), (left_shoulder.shape[1], int(slope*left_shoulder.shape[1] + intercept)), (0, 0, 255), 2)
                else:
                    cv2.line(left_shoulder, (0, int(intercept)), (left_shoulder.shape[1], int(slope*left_shoulder.shape[1] + intercept)), (255, 0, 0), 2)

            #****************************************************************
            #*****This sections out the right shoulder of the subject.*******
            #************repeat same thing for right shoulder****************
            x2 = x - w
            y2 = y + h
            #I used the command below to try to locate the shoulder regions.
                #cv2.rectangle (img, (x2, y2), (x , y + 2*h), (0, 255, 0), 2 )

            right_shoulder = frame[y2 : y + 2*h, x2 : x]
            right_gray = cv2.cvtColor(right_shoulder, cv2.COLOR_BGR2GRAY)
            right_edges = cv2.Canny(right_gray, 100, 200)
            
            right_points = []
            for x_row_r in range (right_edges.shape[1]):
                for y_column_r in range(right_edges.shape[0]):
                    if right_edges[y_column_r, x_row_r] > 0:
                        right_points.append((x_row_r, y_column_r))

            right_x = np.array([p[0] for p in right_points])  
            right_y = np.array([p[1] for p in right_points])

            #I used the slope to check if shrugging was present. The color will be blue if there's
            #no shrugging. It will change to red if there is shrugging.
            if len(right_x) != 0 and len(right_y) != 0:
                slope, intercept = np.polyfit (right_x, right_y , 1)
                if -1.5 <= slope <= -0.3:
                    cv2.line(right_shoulder, (0, int(intercept)), (right_shoulder.shape[1], int(slope*right_shoulder.shape[1] + intercept)), (0, 0, 255), 2)
                else:
                    cv2.line(right_shoulder, (0, int(intercept)), (right_shoulder.shape[1], int(slope*right_shoulder.shape[1] + intercept)), (255, 0, 0), 2)

    #Writes the video frame by frame
    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()