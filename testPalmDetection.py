import sys 
from PIL import Image  
# import the AI model
from modules.openvino_palm_detection import InferenceModel 
from modelplace_api import Device
import cv2
import os 
import math
model = InferenceModel()  # Initialize a model
model.model_load(Device.gpu)  # Loading a model weights
source = 4 
video = cv2.VideoCapture('/home/pcwork/Videos/Webcam/2021-10-07-120459.webm' )
while(True):
    check, frame = video.read()
    #image = Image.open(path_to_image).convert("RGB")  # Read an image
    #image = cv2.imread(path_to_image) 
    image = frame.copy()

    ret = model.process_sample(image)  # Processing an image
    if len(ret) > 0: 
        for i, hand in enumerate(ret):
            if hand.bbox.score >= 0.8:
                distance = math.sqrt( ((hand.bbox.x1-hand.bbox.x2)**2)+((hand.bbox.y1-hand.bbox.y2)**2) )
                w=distance*(150/90)
                h=w
                print ( distance )
                cv2.rectangle(frame,(int(hand.bbox.x1-w/2),int(hand.bbox.y1-h/2)),(int(hand.bbox.x2+w/2),int(hand.bbox.y2+h/2)),(0,0,255),thickness=1)

                for j, point in enumerate(hand.keypoints):
                    cv2.circle(frame,(point.x,point.y),1,(255,0,0), thickness=1)
                    cv2.putText(frame,f"{j}",(point.x,point.y),cv2.FONT_HERSHEY_COMPLEX,fontScale=0.5,color=(255,0,0), thickness=1)

                #print ( hand.keypoints[0] )
    cv2.imshow('image',frame)
    key = cv2.waitKey(1) 
    if key == ord("q"):
        cv2.destroyAllWindows() 
        os._exit(0)



