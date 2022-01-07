from pylab import imshow                                                                     
import numpy as np
import cv2
import torch
import albumentations as albu
from modules.people_segmentation.people_segmentation.pre_trained_models import create_model
from modules.iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from modules.iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
import trimap_module 


import os                                                                             
import time 
workingDir= "./../"  
import torchvision
from modules.RobustVideoMatting.model import MattingNetwork
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor
from PIL import Image





class UnetSegmentation(): 
    def __init__(self,background, pathModel=None): 
        
        self.pathModel = pathModel 
        self.background  = background
        self.model = create_model("Unet_2020-07-20")
        self.model.eval()
        print(self.model)

    def imageMatting(self, source):
        image = cv2.cvtColor(source,cv2.COLOR_BGR2RGB)
        transform = albu.Compose([albu.Normalize(p=1)], p=1)
        padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
        x = transform(image=padded_image)["image"]
        x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
        with torch.no_grad():
          prediction = self.model(x)[0][0]
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
        
        center = self.background.shape 
        x = center[1]/2 - w/2 
        y = center[0]/2 - h/2 
        self.background = self.background[int(y):int(y+h),int(x):int(x+w)]
        #background = background*(1-alpha)
        foregroud = source*alpha
        image = alpha*foregroud+(1-alpha)*self.background
        #cv2.imshow("background",alpha*255)
        #cv2.imshow("foreground",foregroud)


        return alpha*255, image  

class robustMatting(): 
    def __init__(self,background, pathModel=None): 
        self.pathModel= pathModel 
        self.background = background 
        self.model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
        self.model.load_state_dict(torch.load('model/robust/rvm_mobilenetv3.pth'))
    def imageMatting(self, source): 
        transform = torchvision.transforms.ToTensor()
        bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.bgr = torch.tensor([.47,
        rec = [None] * 4                                       # Initial recurrent states.
        downsample_ratio = 0.25                                # Adjust based on your video.

        src = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        src = Image.fromarray(src)
        src = transform(src)
        src = torch.unsqueeze(src, 0)

        with torch.no_grad():
                                # RGB tensor normalized to 0 ~ 1.
            fgr, pha, *rec = self.model(src.cuda(), *rec, downsample_ratio)  # Cycle the recurrent states.
            #fgr, pha, *rec = model(src, *rec, downsample_ratio)  # Cycle the recurrent states.
            com = fgr * pha + bgr * (1 - pha)              # Composite to green background. 
            #writer.write(com)
            img = to_pil_image(com[0])
            img = np.array(img)
            matting = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            alpha = to_pil_image(pha[0])
            alpha = np.array(alpha)
            alpha = cv2.cvtColor(alpha, cv2.COLOR_RGB2BGR)

            return alpha, matting 





background = cv2.imread('background/2.jpg')
source = cv2.imread('/home/pcwork/Desktop/ben.jpg')
#
#ben = UnetSegmentation(background) 
ben = robustMatting(background) 
alpha, matting = ben.imageMatting(source) 
cv2.imwrite("Ben.png", matting)



