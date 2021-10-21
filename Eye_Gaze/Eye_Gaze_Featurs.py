import cv2
import csv
import math
import time
import wave, os, glob


from numpy.core.records import format_parser
from gaze_tracking import GazeTracking
import pandas as pd
import numpy as np

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
data_list = []
#cap=cv2.VideoCapture('non cheating.mp4')
#def getFrame(sec):
       #cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
       #hasFrames,image = cap.read()
       #return hasFrames,image

K = [6.2500000000000000e+002, 0.0, 3.1250000000000000e+002,
     0.0, 6.2500000000000000e+002, 3.1250000000000000e+002,
     0.0, 0.0, 1.0]

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left cornerq
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner

])
df = pd.DataFrame(data_list, columns=['time','blink','right','left','quadrant','left_pupil_x','left_pupil_y','right_pupil_x','right_pupil_y','gaze_center_x', 'gaze_center_y', 'nose_end_points_x','nose_end_points_y','gaze_end_points_x','gaze_end_points_y']
                                      )
df.to_csv("zarin.csv")


sec=0
frameRate = 1
count=1
#success,frame = getFrame(sec)
#while (cap.isOpened):
path = 'zarin'
for f in glob.glob(os.path.join(path, '*.jpg')):
    
    
    frame=cv2.imread(f) #taking frames
    

    size = frame.shape #knowing the size of the frame

    count = count+1
    sec = sec +frameRate
    sec = round(sec,2)
    
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame) #gaze is an instance of gaze tracking class
    blink=0
    right=0
    left=0
    cen=0


    frame = gaze.annotated_frame()
    text = ""
    if gaze.is_blinking():
        text = "Blinking"
        blink=1
    elif gaze.is_right():
        text = "Looking right"
        right=1
    elif gaze.is_left():
        text = "Looking left"
        left=1
    elif gaze.is_center():
        text = "Looking center"
        cen=1
                



    h, w, c = frame.shape
                
        
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
        


    if left_pupil is not None and right_pupil is not None:
                   

        center_x = int((left_pupil[0] + right_pupil[0]) / 2)
        center_y = int((left_pupil[1] + right_pupil[1]) / 2)
               
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)

                    # We are then accesing the landmark points
        i = [33, 8, 36, 45, 48,
                        54]  # Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, right mouth corner
        image_points = []
        for n in i:
             x = gaze.gaze_landmarks.part(n).x
             y = gaze.gaze_landmarks.part(n).y;
                       
             image_points += [(x, y)]
             cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

        image_points = np.array(image_points, dtype="double")
                   
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                                cam_matrix, dist_coeffs,
                                                                                flags=cv2.SOLVEPNP_ITERATIVE)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                                    translation_vector,
                                                                    cam_matrix, dist_coeffs)

        for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                center_nose_x = image_points[0][0]
                center_nose_y = image_points[0][1]
                end_nose_x = nose_end_point2D[0][0][0]
                end_nose_y = nose_end_point2D[0][0][1]

        if 0 < end_nose_x < center_nose_x and 0 < end_nose_y < center_nose_y:
                        data_list.append([sec,blink,right,left,1, left_pupil[0],left_pupil[1], right_pupil[0],right_pupil[1], center_x, center_y, p1[0],p1[1], p2[0],p2[1]])

        elif center_nose_x < end_nose_x < (w*10) and 0 < end_nose_y < center_nose_y:
                        data_list.append([sec,blink,right,left,2, left_pupil[0],left_pupil[1], right_pupil[0],right_pupil[1], center_x, center_y, p1[0],p1[1], p2[0],p2[1]])

        elif 0 < end_nose_x < center_nose_x and center_nose_y < end_nose_y < (h*10):
                        data_list.append([sec,blink,right,left,4, left_pupil[0],left_pupil[1], right_pupil[0],right_pupil[1], center_x, center_y, p1[0],p1[1], p2[0],p2[1]])
        else:

                        data_list.append([sec,blink,right,left,3, left_pupil[0],left_pupil[1], right_pupil[0],right_pupil[1], center_x, center_y, p1[0],p1[1], p2[0],p2[1]])


        cv2.line(frame, (0, int(center_nose_y)), (w, int(center_nose_y)), (0, 255, 0), 2)
        cv2.line(frame, (int(center_nose_x), 0), (int(center_nose_x), h), (0, 255, 0), 2)
        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 1)

        cv2.imshow("Gaze Tracking window", frame)
        with open('zarin.csv','w',newline='') as t:
                        fieldnames=['time','blink','right','left','quadrant','left_pupil_x','left_pupil_y','right_pupil_x','right_pupil_y','gaze_center_x', 'gaze_center_y', 'nose_end_points_x','nose_end_points_y','gaze_end_points_x','gaze_end_points_y']
                        thewriter=csv.DictWriter(t,fieldnames=fieldnames)
                        thewriter.writeheader()
                        thewriter=csv.writer(t)
                        thewriter.writerows(data_list)

        if cv2.waitKey(1) == 27:
                        break
    
    else:
    
        data_list.append([sec,blink,right,left,0,0,0,0,0,0,0,0,0,0,0])

        with open('zarin.csv','w',newline='') as t:
                        fieldnames=['time','blink','right','left','quadrant','left_pupil_x','left_pupil_y','right_pupil_x','right_pupil_y','gaze_center_x', 'gaze_center_y', 'nose_end_points_x','nose_end_points_y','gaze_end_points_x','gaze_end_points_y']
                        thewriter=csv.DictWriter(t,fieldnames=fieldnames)
                        thewriter.writeheader()
                        thewriter=csv.writer(t)
                        thewriter.writerows(data_list) 
    
                  





           
