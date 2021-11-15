import os  
import time
import numpy as np 
workingDir= "./../" 


p="/home/pcwork/ai/ftech/finger/handGestureWithMatting/evaluation/archive/Achieve"
path = os.listdir(f"{p}/image")
path.sort()

IoUScore = np.array([])
timeRunning = np.array([])




from pylab import imshow
import numpy as np
import cv2
import torch
import albumentations as albu
from modules.people_segmentation.people_segmentation.pre_trained_models import create_model
from modules.iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from modules.iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
import trimap_module 
#if torch.cuda.is_available():
#    device = torch.device("cuda")
#    model = create_model("Unet_2020-07-20")
#else: 
model = create_model("Unet_2020-07-20")
#model.to(torch.device("cuda"))
model.eval()
print ( model)
background = cv2.imread('background/2.jpg') 

#background = cv2.imread('ftech.jpeg')

video = '/home/pcwork/Videos/Webcam/2021-09-04-113212.webm'  
count = 0 
cam = cv2.VideoCapture(video) 
for i, pathImg in enumerate(path): 
    imgSource = cv2.imread(f"{p}/image/{pathImg}")
    #imgTrue = cv2.imread(f"{p}PedMasks/{pathImg[:-4]}_mask.png")     
    imgTrue = cv2.imread(f"{p}/alpha/{pathImg[:-4]}.png",0)
    timeBegin = time.time() 

    frame = imgSource.copy()
    #image = load_rgb("/home/pcwork/Desktop/ben.PNG")
    
    image = cv2.cvtColor(imgSource,cv2.COLOR_BGR2RGB)
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
    with torch.no_grad():
      prediction = model(x)[0][0]
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)
    imshow(mask)
    dst = cv2.addWeighted(image, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0)
    image = cv2.cvtColor(dst,cv2.COLOR_RGB2BGR) 
    alpha = mask 
    #alpha = trimap_module.trimap(alpha*255,size=5,erosion=False)
    #cv2.imshow("alpha",alpha)
    print ( alpha.shape)
    alpha = cv2.cvtColor(alpha,cv2.COLOR_GRAY2BGR) 
    h,w,_ = alpha.shape

    center = background.shape 
    x = center[1]/2 - w/2 
    y = center[0]/2 - h/2 
    background = background[int(y):int(y+h),int(x):int(x+w)]
    #background = background*(1-alpha)
    foregroud = frame*alpha
    image = alpha*foregroud+(1-alpha)*background
    cv2.imwrite(f"image/my{count}.jpg",alpha*255)

    imgOut = cv2.cvtColor(alpha,cv2.COLOR_BGR2GRAY)*255 

    timeRunning= np.append(timeRunning,time.time()-timeBegin)
    intersection = np.logical_and(imgTrue, imgOut)
    union = np.logical_or(imgTrue, imgOut)
    iou_score = np.sum(intersection) / np.sum(union)
    IoUScore = np.append(IoUScore, iou_score)
    print(iou_score)

    count +=1 
    cv2.imshow("background",alpha*255)
    cv2.imshow("foreground",foregroud)
    cv2.imshow("image",image)
    cv2.imshow("imgTrue",imgTrue)
    
    if i == 400:
        break 
    key = cv2.waitKey(1)
    if key == ord('q'): 
        break 

print ("result")    
print ( timeRunning.mean() )
print ( IoUScore.mean() )



cv2.destroyAllWindows
cam.release()
