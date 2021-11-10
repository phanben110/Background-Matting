import cvzone 
from cvzone.SelfiSegmentationModule import SelfiSegmentation 
import os  
import cv2

imgBg = cv2.imread('/home/pcwork/ai/ftech/finger/handGestureWithMatting/background/3.jpg' )
imgBg = cv2.resize(imgBg,(640,480))
cap = cv2.VideoCapture('/home/pcwork/Videos/Webcam/2021-09-04-113212.webm')
#cap = cv2.VideoCapture(0)
segmentor = SelfiSegmentation() 

while True: 
    success, img = cap.read() 
    imgOut= segmentor.removeBG(img, imgBg, threshold=0.5)
    cv2.imshow("image",imgOut )
    key = cv2.waitKey(1)
    if key == ord('q'):
        break 
cv2.destroyAllWindows()
