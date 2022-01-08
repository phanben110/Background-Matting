#import some library for segmentation
from pylab import imshow                                                                     
import numpy as np
import cv2
import torch
import albumentations as albu
from modules.people_segmentation.people_segmentation.pre_trained_models import create_model
from modules.iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from modules.iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
import trimap_module 

#import some library for robust matting 
import os                                                                             
import time 
workingDir= "./../"  
import torchvision
from modules.RobustVideoMatting.model import MattingNetwork
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor
from PIL import Image

import torch.nn.functional as F
import torchvision.transforms as transforms
from modules.MODNet.src.models.modnet import MODNet




from skimage.transform import resize
from torchvision import transforms
import logging
from modules.P3M.core.config import *
from modules.P3M.core.util import *
from modules.P3M.core.evaluate import *
from modules.P3M.core.network.P3mNet import P3mNet

def resizeBacground(matte, background):
    h,w,_ = matte.shape
    center = background.shape 
    if center[0] >=h and center[1] >=w:
        x = center[1]/2 - w/2 
        y = center[0]/2 - h/2 
        background = background[int(y):int(y+h),int(x):int(x+w)]
    else: 
        background = cv2.resize(background,(w,h))
    return background


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
        
        self.background = resizeBacground(alpha, self.background)
        #background = background*(1-alpha)
        foregroud = source*alpha
        image = alpha*foregroud+(1-alpha)*self.background
        #cv2.imshow("background",alpha*255)
        #cv2.imshow("foreground",foregroud)


        return alpha*255, image  

class robustMatting(): 
    def __init__(self,background, pathModel=None): 
        self.pathModel= pathModel 
        self.background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)  
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

            alpha = to_pil_image(pha[0])
            pha = np.array(alpha)
            alpha = cv2.cvtColor(pha, cv2.COLOR_RGB2BGR)

            fgr = to_pil_image(fgr[0]) 
            fgr = np.array(fgr)      

            #fgr = fgr.convert('RGB')
            #pha = pha.convert('L')

            self.background = resizeBacground(alpha, self.background)

            pha = np.asarray(pha).astype(float)[:, :, None] / 255
            matting = Image.fromarray(np.uint8(np.asarray(fgr) * pha + np.asarray(self.background) * (1 - pha)))

            matting = np.array(matting)

            matting = cv2.cvtColor(matting, cv2.COLOR_RGB2BGR)
            
            return alpha, matting 


class mediapipe(): 
    def __init__(self,background, pathModel = None ): 
        self.pathModel = pathModel 
        self.background = background
        import cvzone
        from BEN_MediapipeMatting import SelfiSegmentation 
        self.model = SelfiSegmentation()
    def imageMatting(self,source): 

        self.background = resizeBacground(source, self.background)
                                                                
        matting, alpha = self.model.removeBG(source, self.background, threshold=0.5)
        #alpha = cv2.cvtColor(alpha, cv2.COLOR_RGB2BGR)
        print (alpha.shape)
        return alpha*255, matting

        
class modNet(): 
    def __init__(self, background, pathModel = None): 
        self.pathModel = pathModel 
        self.background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)  


        self.ref_size = 512       
                     
        # define image to tensor transform
        self.im_transform = transforms.Compose(
            [                
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]                
        )                    
                             
        # create MODNet and load the pre-trained ckpt
        self.model = MODNet(backbone_pretrained=False)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.load_state_dict(torch.load('model/modnet/modnet_webcam_portrait_matting.ckpt'))                 
        self.model.eval()

    def combinedDisplay (self, image, matte): 
        # calculate display resolution

        w, h, _  = image.shape
        rw, rh = 800, int(h * 800 / (3 * w))
        
        # obtain predicted foreground
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]
        matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
         
        self.background = resizeBacground(matte, self.background)

        
        #foreground = image * matte + np.full(image.shape, 255) * (1 - matte)
        foreground = image * matte + self.background * (1 - matte)
        
        # combine image, foreground, and alpha into one line
        #combined = np.concatenate((image, foreground, matte * 255), axis=1)
        combined = Image.fromarray(np.uint8(foreground))
        combined = np.array(combined)
        return combined

    def imageMatting(self,source): 

        # inference images
        im = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        image = im.copy()

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]
        
        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = self.im_transform(im)
        
        # add mini-batch dim
        im = im[None, :, :, :]
        
        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < self.ref_size or min(im_h, im_w) > self.ref_size:
            if im_w >= im_h:
                im_rh = self.ref_size
                im_rw = int(im_w / im_h * self.ref_size)
            elif im_w < im_h:
                im_rw = self.ref_size
                im_rh = int(im_h / im_w * self.ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
        
        # inference
        _, _, matte = self.model(im.cuda(), True)
        
        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte = Image.fromarray(((matte * 255).astype('uint8')), mode='L')
        matting = self.combinedDisplay(image, matte)
        return np.array(matte), matting 


class P3MNet():
    def __init__(self,background): 
        self.background = background  
        self.pathModel = 'modules/P3M/models/pretrained/p3mnet_pretrained_on_p3m10k.pth' 
        self.model = P3mNet()
        if torch.cuda.device_count()==0:
            print(f'Running on CPU...')
            ckpt = torch.load(self.pathModel, map_location=torch.device('cpu'))
            self.cuda= False 
        else:
            self.cuda = True
            print(f'Running on GPU with CUDA as {self.cuda}...')
            ckpt = torch.load(self.pathModel)
        self.model.load_state_dict(ckpt['state_dict'], strict=True)
        if self.cuda:
            self.model = self.model.cuda()
        self.model.eval()

            #test_samples(args,model)


    def inference_once(self,scale_img, scale_trimap=None):
        pred_list = []
        if torch.cuda.is_available():
            tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1).cuda()
        else:
            tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1)
        input_t = tensor_img
        input_t = input_t/255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        input_t = normalize(input_t)
        input_t = input_t.unsqueeze(0)
        pred_global, pred_local, pred_fusion = self.model(input_t)[:3]
        pred_global = pred_global.data.cpu().numpy()
        pred_global = gen_trimap_from_segmap_e2e(pred_global)
        pred_local = pred_local.data.cpu().numpy()[0,0,:,:]
        pred_fusion = pred_fusion.data.cpu().numpy()[0,0,:,:]        
        return pred_global, pred_local, pred_fusion

    def imageMatting(self, source): 
        image = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        img = np.array(image)[:,:,:3]

        h, w, c = img.shape

        if min(h, w)>SHORTER_PATH_LIMITATION:
          if h>=w:
              new_w = SHORTER_PATH_LIMITATION
              new_h = int(SHORTER_PATH_LIMITATION*h/w)
              img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
          else:
              new_h = SHORTER_PATH_LIMITATION
              new_w = int(SHORTER_PATH_LIMITATION*w/h)
              img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        with torch.no_grad():
            if self.cuda:
                torch.cuda.empty_cache()
            alpha = self.inference_img(img)

        #composite = generate_composite_img(img, predict)
        #alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = alpha*255.0

        alpha = alpha.astype('float32')
        
        alpha = cv2.cvtColor(alpha,cv2.COLOR_GRAY2BGR)/255
        self.background = resizeBacground(alpha, self.background)
        matting = alpha*source + (1-alpha)*self.background
        matting = matting.astype('float32')
        cv2.imwrite("alpha.png", alpha*255) 
        cv2.imwrite("matting.png", matting)
        cv2.imwrite("source.png", source)

        return alpha*255.0, matting



    def inference_img(self,img) :
        h, w, c = img.shape
        new_h = min(MAX_SIZE_H, h - (h % 32))
        new_w = min(MAX_SIZE_W, w - (w % 32))
        
        resize_h = int(h/2)
        resize_w = int(w/2)
        new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
        new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
        scale_img = resize(img,(new_h,new_w))*255.0
        pred_global, pred_local, pred_fusion = self.inference_once(scale_img)
        #pred_local = resize(pred_local,(h,w))
        #pred_global = resize(pred_global,(h,w))*255.0
        #cv2.imwrite("pred_global.png",pred_global)
        #cv2.imwrite("alpha.png",pred_global*pred_local*255)
        #cv2.imwrite("pred_fusion.png",pred_local*255)
        pred_fusion = resize(pred_fusion,(h,w))
        return pred_fusion





background = cv2.imread('/home/pcwork/ai/ftech/finger/handGestureWithMatting/background/0.jpg'  )
source = cv2.imread('/home/pcwork/ai/ftech/finger/handGestureWithMatting/modules/P3M/samples/original/p_015cd10e.jpg'  )
#
ben = P3MNet(background) 
alpha, matting = ben.imageMatting(source) 

#image = cv2.add(matting, background)

cv2.imwrite("Ben222.png", matting)
cv2.imwrite("alpha1.png", alpha)

#ben = robustMatting(background) 
#alpha, matting = ben.imageMatting(source) 
#cv2.imwrite("Ben222.png", matting)
#ben = mediapipe(background) 
#alpha, matting = ben.imageMatting(source) 
#ben = modNet(background) 
#alpha, matting = ben.imageMatting(source) 



