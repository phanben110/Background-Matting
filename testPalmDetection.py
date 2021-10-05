import sys 
from PIL import Image  
# import the AI model
from modules.openvino_palm_detection import InferenceModel 
from modelplace_api import Device
import cv2
import os 

model = InferenceModel()  # Initialize a model
model.model_load(Device.cpu)  # Loading a model weights
source = 4 
video = cv2.VideoCapture(source)
while(True):
    check, frame = video.read()
    #image = Image.open(path_to_image).convert("RGB")  # Read an image
    #image = cv2.imread(path_to_image) 
    image = frame.copy()

    ret = model.process_sample(image)  # Processing an image
    if len(ret) > 0: 
        for i, hand in enumerate(ret):
            if hand.bbox.score >= 0.8:
                bouding = [hand.bbox.x1,hand.bbox.y1,hand.bbox.x2,hand.bbox.y2] 
                cv2.rectangle(frame,(hand.bbox.x1,hand.bbox.y1),(hand.bbox.x2,hand.bbox.y2),(0,0,255),thickness=1)
                for j, point in enumerate(hand.keypoints):
                    cv2.circle(frame,(point.x,point.y),1,(255,0,0), thickness=1)
                #print ( hand.keypoints[0] )
    cv2.imshow('image',frame)
    key = cv2.waitKey(1) 
    if key == ord("q"):
        cv2.destroyAllWindows() 
        os._exit(0)



