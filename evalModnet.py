import os                                                                                         
import time 
import numpy as np  
workingDir= "./../"  
 
 
p="/home/pcwork/ai/ftech/finger/handGestureWithMatting/evaluation/archive/Achieve"
path = os.listdir(f"{p}/image")
path.sort() 
 
IoUScore = np.array([]) 
timeRunning = np.array([]) 








import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from modules.MODNet.src.models.modnet import MODNet


torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

print('Load pre-trained MODNet...')
pretrained_ckpt = 'model/modnet/modnet_webcam_portrait_matting.ckpt'
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

GPU = True if torch.cuda.device_count() > 0 else False
GPU = True 
if GPU:
    print('Use GPU...')
    modnet = modnet.cuda()
    modnet.load_state_dict(torch.load(pretrained_ckpt))
else:
    print('Use CPU...')
    modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))

modnet.eval()

print('Init WebCam...')
cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/home/pcwork/Videos/Webcam/2021-09-04-113212.webm'  )

#cap = cv2.VideoCapture('video.mp4'  )
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280 )
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

print('Start matting...')
#while(True):
for i, pathImg in enumerate(path): 
    imgSource = cv2.imread(f"{p}/image/{pathImg}")
    #imgTrue = cv2.imread(f"{p}PedMasks/{pathImg[:-4]}_mask.png")   
    imgTrue = cv2.imread(f"{p}/alpha/{pathImg[:-4]}.png",0)
    timeBegin = time.time() 

    frame = imgSource.copy()
    #image = load_rgb("/home/pcwork/Desktop/ben.PNG")

    imgTrue = cv2.resize(imgTrue, (672,512))

    #_, frame_np = cap.read()
    frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
    frame_np = frame_np[:, 120:792, :]
    frame_np = cv2.flip(frame_np, 1)

    frame_PIL = Image.fromarray(frame_np)
    frame_tensor = torch_transforms(frame_PIL)
    frame_tensor = frame_tensor[None, :, :, :]
    print ( f"shape frame tensor {frame_tensor.shape}"  )
    if GPU:
        frame_tensor = frame_tensor.cuda()
    
    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_tensor, True)

    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
    fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
    view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))
    view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)
    #alpha = np.uint8(matte_np)
    alpha = matte_np.copy()
    alpha = cv2.cvtColor(alpha, cv2.COLOR_RGB2BGR)
    imgOut = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)*255

    
    timeRunning= np.append(timeRunning,time.time()-timeBegin)
    intersection = np.logical_and(imgTrue, imgOut)
    union = np.logical_or(imgTrue, imgOut)
    iou_score = np.sum(intersection) / np.sum(union)
    IoUScore = np.append(IoUScore, iou_score)
    print(iou_score)

    print (f"out shape {alpha.shape} ")
    cv2.imshow ("alpha", alpha*255)
    cv2.imshow ("imgTrue", imgTrue)


    cv2.imshow('MODNet - WebCam [Press \'Q\' To Exit]', view_np)
    if i == 500: 
        break 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print ("result")    
print ( timeRunning.mean() )
print ( IoUScore.mean() )    


print('Exit...')
