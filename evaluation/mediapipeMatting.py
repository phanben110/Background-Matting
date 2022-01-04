import cvzone 
import os   
import cv2 
import time

from cvzone.SelfiSegmentationModule import SelfiSegmentation 
import os  
import numpy as np 
import cv2
p="/home/pcwork/ai/ftech/finger/handGestureWithMatting/evaluation/archive/Achieve"
path = os.listdir(f"{p}/image")
path.sort()

imgBg = cv2.imread('/home/pcwork/ai/ftech/finger/handGestureWithMatting/background/3.jpg' )
imgBg = cv2.resize(imgBg,(800,600))
cap = cv2.VideoCapture('/home/pcwork/Videos/Webcam/2021-09-04-113212.webm')
#cap = cv2.VideoCapture(0)
segmentor = SelfiSegmentation() 
imgBg = np.zeros((800, 600, 1), dtype = "uint8")
IoUScore = np.array([])
timeRunning = np.array([])
import numpy

for i, pathImg in enumerate(path): 
    imgSource = cv2.imread(f"{p}/image/{pathImg}")
    #imgTrue = cv2.imread(f"{p}PedMasks/{pathImg[:-4]}_mask.png")
    imgTrue = cv2.imread(f"{p}/alpha/{pathImg[:-4]}.png",0)
    timeBegin = time.time() 
    #print ( f"{p}/alpha/{pathImg[:-3]}.png")
    imgOut= segmentor.removeBG(imgSource, imgBg, threshold=0.5)
     #alpha = (img-imgBg)/(imgOut-imgBg)
    imgOut =cv2.cvtColor(imgOut,cv2.COLOR_BGR2GRAY)
    (thresh, imgOut) = cv2.threshold(imgOut, 1, 255, cv2.THRESH_BINARY)
    #cv2.imshow("alpha",alpha*255)
    #cv2.imshow("image",imgOut )
    #cv2.imwrite("image.png",alpha*255)
    timeRunning= np.append(timeRunning,time.time()-timeBegin)
    intersection = numpy.logical_and(imgTrue, imgOut)
    union = numpy.logical_or(imgTrue, imgOut)
    iou_score = numpy.sum(intersection) / numpy.sum(union)
    IoUScore = np.append(IoUScore, iou_score)
    print(iou_score)




    cv2.imshow("imgSource", imgSource)
    cv2.imshow("imgTrue",imgTrue)
    key = cv2.waitKey(1)
    #time.sleep(0.3)
    if i == 400: 
        break
    if key == ord('q'):
        break 




print ( timeRunning.mean() )
print ( IoUScore.mean() )
cv2.destroyAllWindows()
