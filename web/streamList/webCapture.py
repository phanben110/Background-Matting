import streamlit as st 
import pandas as pd 
import cv2 

cam = cv2.VideoCapture(0) 
while(1):
    _, frame = cam.read() 
    #cv2.imshow ("ben dep trai", frame) 
    st.image(frame,caption="ben dep trai")

    key = cv2.waitKey(1)
    if key == ord("q"): 
        break 
cam.release()
cv2.destroyAllWindows()
