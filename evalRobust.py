import os                                                                                 
import time 
import numpy as np  
workingDir= "./../"  
 
 
p="/home/pcwork/ai/ftech/finger/handGestureWithMatting/evaluation/archive/Achieve"
path = os.listdir(f"{p}/image")
path.sort() 
 
IoUScore = np.array([]) 
timeRunning = np.array([]) 








import torchvision
from modules.RobustVideoMatting.model import MattingNetwork
from PIL import Image  
import cv2  
from torchvision.transforms.functional import to_pil_image
import numpy as np 
import torch

model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('model/robust/rvm_mobilenetv3.pth'))
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
#from inference_utils import VideoReader, VideoWriter, ImageSequenceReader , ImageSequenceWriter

#reader = VideoReader('/home/pcwork/Videos/Webcam/2021-09-04-113212.webm' , transform=ToTensor())
source  = '/home/pcwork/Videos/Webcam/2021-09-04-113212.webm' 
#source = 0 
cap = cv2.VideoCapture(source) 
bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1) # Green background.
rec = [None] * 4                                       # Initial recurrent states.
downsample_ratio = 0.25                                # Adjust based on your video.

transform = torchvision.transforms.ToTensor()

for i, pathImg in enumerate(path): 
    imgSource = cv2.imread(f"{p}/image/{pathImg}")
    #imgTrue = cv2.imread(f"{p}PedMasks/{pathImg[:-4]}_mask.png")   
    imgTrue = cv2.imread(f"{p}/alpha/{pathImg[:-4]}.png",0)
    timeBegin = time.time() 

    frame = imgSource.copy()
    #image = load_rgb("/home/pcwork/Desktop/ben.PNG")



    #reader = ImageSequenceReader('image' ,transform=ToTensor())
    src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    src = Image.fromarray(src)
    src = transform(src)
    src = torch.unsqueeze(src, 0)
    print ( src.shape )

    #writer = VideoWriter('Phan Ben.mp4', frame_rate=30)
    #writer = ImageSequenceWriter("ben")
    
    with torch.no_grad():
                            # RGB tensor normalized to 0 ~ 1.
        fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio)  # Cycle the recurrent states.
        #fgr, pha, *rec = model(src, *rec, downsample_ratio)  # Cycle the recurrent states.
        com = fgr * pha + bgr * (1 - pha)              # Composite to green background. 
        #writer.write(com)
        img = to_pil_image(com[0])
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        alpha = to_pil_image(pha[0])
        alpha = np.array(alpha)
        imgOut = cv2.cvtColor(alpha, cv2.COLOR_RGB2BGR)
        imgOut = cv2.cvtColor(imgOut, cv2.COLOR_BGR2GRAY)


        #cv2.imwrite("bendeptrai.jpg",img)
        # Write frame.



    timeRunning= np.append(timeRunning,time.time()-timeBegin)
    intersection = np.logical_and(imgTrue, imgOut)
    union = np.logical_or(imgTrue, imgOut)
    iou_score = np.sum(intersection) / np.sum(union)
    IoUScore = np.append(IoUScore, iou_score)
    print(iou_score)

    cv2.imshow("demo",img) 
    cv2.imshow("alpha", imgOut) 
    cv2.imshow("imageTrue", imgTrue)
    key = cv2.waitKey(1) 
    if i == 400: 
        break 
    if key == ord("q"): 
        break 

print ("result")    
print ( timeRunning.mean() )
print ( IoUScore.mean() )


cv2.destroyAllWindows()
