import torchvision
from model import MattingNetwork
from PIL import Image  
import cv2  
from torchvision.transforms.functional import to_pil_image
import numpy as np 
import torch

model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from inference_utils import VideoReader, VideoWriter, ImageSequenceReader , ImageSequenceWriter

#reader = VideoReader('/home/pcwork/Videos/Webcam/2021-09-04-113212.webm' , transform=ToTensor())
source  = '/home/pcwork/Videos/Webcam/2021-09-04-113212.webm' 
cap = cv2.VideoCapture(source) 
bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1) # Green background.
rec = [None] * 4                                       # Initial recurrent states.
downsample_ratio = 0.25                                # Adjust based on your video.

transform = torchvision.transforms.ToTensor()
while True: 
    _, frame = cap.read() 
    
    reader = ImageSequenceReader('image' ,transform=ToTensor())
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
        print ( com[0].shape)
        img = to_pil_image(com[0])
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        print ( img.shape )
        #cv2.imwrite("bendeptrai.jpg",img)
        # Write frame.
    cv2.imshow("demo",img) 
    key = cv2.waitKey(1) 
    if key == ord("q"): 
        break 
cv2.destroyAllWindows()
