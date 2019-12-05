# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:41:41 2019

@author: Lenovo
"""

import cv2
import numpy as np
import dlib
from math import hypot

font=cv2.FONT_HERSHEY_SIMPLEX
cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def mid_pt(p1,p2):
    return int((p1.x+p2.x)/2),int((p1.y+p2.y)/2)

def eye_ratio(eye_pos,eye_landmark):
        lft_pt=(eye_landmark.part(eye_pos[0]).x,eye_landmark.part(eye_pos[0]).y)
        rgt_pt=(eye_landmark.part(eye_pos[3]).x,eye_landmark.part(eye_pos[3]).y)
        top_pt=mid_pt(eye_landmark.part(eye_pos[1]),eye_landmark.part(eye_pos[2]))
        btm_pt=mid_pt(eye_landmark.part(eye_pos[5]),eye_landmark.part(eye_pos[4]))
        
        hor_len=hypot((lft_pt[0]-rgt_pt[0]),(lft_pt[1]-rgt_pt[1]))
        ver_len=hypot((top_pt[0]-btm_pt[0]),(top_pt[1]-btm_pt[1]))
        
        #cv2.line(img,lft_pt,rgt_pt,(0,255,0),2)
        #cv2.line(img,top_pt,btm_pt,(0,255,0),2)
        
        len_ratio=hor_len/ver_len
        return len_ratio
    
    
while 1:
    _,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        landmark=predictor(gray,face)
        left_eye=eye_ratio([36,37,38,39,40,41],landmark)
        right_eye=eye_ratio([42,43,44,45,46,47],landmark)
        ratio=(left_eye+right_eye)/2
        if ratio>4.8:
            cv2.putText(img,'BLINKING',(100,100),font,2,(0,0,255),3)
        
    cv2.imshow('Reading',img)
    
    k=cv2.waitKey(5) & 0xff
    if k==27:
        break
    
cap.release()
cv2.destroyAllWindows()