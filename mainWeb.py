import streamlit as st
import os 
import cv2
import tempfile
from BEN_MattingMethod import UnetSegmentation as unet 
from BEN_MattingMethod import robustMatting as robust 
from BEN_MattingMethod import mediapipe 
from BEN_MattingMethod import modNet
from BEN_MattingMethod import P3MNet 
from BEN_footer import footer
from evaluate import * 

optionOutput = []

#st.title('BACKGROUND MATTING')
#st.markdown(f"<h1 style='text-align: center; color: blue;'>Background Matting</h1>", unsafe_allow_html=True)
#st.set_page_config(page_title="BenPhan", page_icon=”🖖”)
st.set_page_config(page_title="Ben Phan", page_icon="🖖")
st.image("ftech3.jpg")





st.sidebar.image("logoFtech.png", use_column_width=True)
optionOutput = st.sidebar.multiselect(
     'Output Display',
     ['Input source', 'Output alpha', 'Output matting'])

print ( optionOutput )

def robustMatting(source, background):
    alpha = None 
    matting = None 
    return source, alpha, matting 

def unetSegmentation(source, background): 
    alpha = None 
    matting = None 
    return source, alpha, matting 

#def modNet(source, background): 
#    alpha = None 
#    matting = None 
#    return source, alpha, matting 

def mediapipeMatting(source, background): 
    alpha = None  
    matting = None 
    return source, alpha, matting 

#def P3MNet (source, background): 
#    alpha = None 
#    matting = None 
#    return source, alpha, matting 

def GFMNet (source, background): 
    alpha = None 
    matting = None  
    return source, alpha, matting 

def chooseMethod (optionMethod,background): 
    if optionMethod == "Unet Segmentation": 
        model = unet(background)
        return model  
    elif optionMethod == "Robust Matting": 
        model = robust(background)
        return model 
    elif optionMethod == "ModNet": 
        model = modNet(background)
        return model  
    elif optionMethod == "Mediapipe Matting": 
        model = mediapipe(background)
        return model 
    elif optionMethod == "P3MNet":
        model = P3MNet(background)
        return model
    elif optionMethod == "GFMNet": 
        return GFMNet(source, background) 


if len(optionOutput) == 3 :            
    col1, col2, col3 = st.columns(3)

    with col1:
        st.text(optionOutput[0])
        #st.markdown(f"<h1 style='text-align: center; color: red;'>{optionOutput[0]}</h1>", unsafe_allow_html=True)
        stframe1 = st.empty()          
    
    with col2:
        st.text(optionOutput[1])
        stframe2 = st.empty()          
    
    with col3:
        st.text(optionOutput[2])
        stframe3 = st.empty()          
elif len(optionOutput) == 2 :
    col1, col2 = st.columns(2) 

    with col1: 
        st.text(optionOutput[0])
        stframe1 = st.empty() 
    with col2: 
        st.text(optionOutput[1])
        stframe2 = st.empty()
else :                    
    try:
        st.text(optionOutput[1])
        col1 = st.columns(1) 
        stframe1 = st.empty()
    except: 
        col1 = st.columns(1) 
        stframe1 = st.empty()




method = st.sidebar.selectbox(
     'Which method do you want to use?',
     ('None','Unet Segmentation','Robust Matting','ModNet','Mediapipe Matting','P3MNet','GFMNet'))

if method != "None":
    st.text(f"* Use the {method} method")


option = st.sidebar.selectbox(
     'How would you like to do?',
     ('Image Matting', 'Video Matting', 'Evaluation'))

background = st.sidebar.file_uploader("Upload background")
tbackground = tempfile.NamedTemporaryFile(delete=False) 

if st.sidebar.checkbox('I agree above setting'):
        
    if background is not None:
        tbackground.write(background.read())                                 
        imgBackground = cv2.imread(tbackground.name)   
    else: 
        imgBackground = cv2.imread('background/2.jpg')
    
    
    if option == "Video Matting":
        optionVideo = st.sidebar.selectbox(
            'Choose source?',
            ('Webcam 0', 'Webcam 1', "Upload Video"))
        source = None 
        if optionVideo == "Webcam 0":
            source = 0 
        elif optionVideo == "Webcam 1":
            source = 4 
        elif optionVideo == "Upload Video":
            f = st.sidebar.file_uploader("Upload video")
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            if f is not None:
                tfile.write(f.read())
                source = tfile.name 
        if source is not None:
    
            cam = cv2.VideoCapture(source)
    
            if st.sidebar.button("Run"):

                model = chooseMethod(method,imgBackground)

                while cam.isOpened():                                                  
                    ret, frame = cam.read()                                            
                    # if frame is read correctly ret is True                          
                    if not ret:                                                       
                        print("Can't receive frame (stream end?). Exiting ...")       
                        cam.release()                                                  
                        break                                                         
                    alpha, matting = model.imageMatting(frame)
                    source = frame.copy()
                    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)                       
                    if method == "Mediapipe Matting" or method == "ModNet":
                        pass 
                    else :
                        alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2RGB)     
                    if method == "ModNet": 
                        pass
                    else:
                        matting = cv2.cvtColor(matting, cv2.COLOR_BGR2RGB)                       
    
                    if len (optionOutput) == 3:
                        i1 = optionOutput[0].index(" ")
                        i2 = optionOutput[1].index(" ")
                        i3 = optionOutput[2].index(" ")
                        frame1 = optionOutput[0][i1+1:]
                        frame2 = optionOutput[1][i2+1:]
                        frame3 = optionOutput[2][i3+1:]
                        if method == "P3MNet":
                            stframe1.image(f"{frame1}.png")          
                            stframe2.image(f"{frame2}.png")          
                            stframe3.image(f"{frame3}.png")          
                        else:
                            stframe1.image(eval(frame1))          
                            stframe2.image(eval(frame2))          
                            stframe3.image(eval(frame3))          

                    elif len ( optionOutput ) == 2 : 
                        i1 = optionOutput[0].index(" ")
                        i2 = optionOutput[1].index(" ")
                        frame1 = optionOutput[0][i1+1:]
                        frame2 = optionOutput[1][i2+1:]
                        if method == "P3MNet":
                            stframe1.image(f"{frame1}.png")          
                            stframe2.image(f"{frame2}.png")          
                        else:
                            stframe1.image(eval(frame1))         
                            stframe2.image(eval(frame2))         

                    elif len (optionOutput) == 1 : 
                        i1 = optionOutput[0].index(" ")
                        frame1 = optionOutput[0][i1+1:]
                        if method == "P3MNet":
                            stframe1.image(f"{frame1}.png")          
                        else:
                            stframe1.image(eval(frame1))         
                    else :
                        if method == "P3MNet":
                            stframe1.image("matting.png")          
                        else:
                            stframe1.image(matting)
                        
                    print ( "done1" )
    
    
    
                                                                          
    
    elif option == "Image Matting": 
        f = st.sidebar.file_uploader("Upload image")
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        if f is not None:
            if st.sidebar.button("Run"):
                model = chooseMethod(method,imgBackground)
                tfile.write(f.read())
                image = cv2.imread(tfile.name)
                source = image.copy()
                alpha, matting = model.imageMatting(image)
                source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)                       
                if method == "Mediapipe Matting" or method =="ModNet":
                    pass 
                else :
                    alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2RGB)                       
                if method == "ModNet": 
                    pass 
                else:
                    matting = cv2.cvtColor(matting, cv2.COLOR_BGR2RGB)                       



                if len (optionOutput) == 3:
                    i1 = optionOutput[0].index(" ")
                    i2 = optionOutput[1].index(" ")
                    i3 = optionOutput[2].index(" ")
                    frame1 = optionOutput[0][i1+1:]
                    frame2 = optionOutput[1][i2+1:]
                    frame3 = optionOutput[2][i3+1:]
                    if method == "P3MNet":
                        stframe1.image(f"{frame1}.png")   
                        stframe2.image(f"{frame2}.png")   
                        stframe3.image(f"{frame3}.png")   
                    else:
                        stframe1.image(eval(frame1))      
                        stframe2.image(eval(frame2))      
                        stframe3.image(eval(frame3))      
                
                elif len ( optionOutput ) == 2 : 
                    i1 = optionOutput[0].index(" ")
                    i2 = optionOutput[1].index(" ")
                    frame1 = optionOutput[0][i1+1:]
                    frame2 = optionOutput[1][i2+1:]
                    if method == "P3MNet":
                        stframe1.image(f"{frame1}.png")   
                        stframe2.image(f"{frame2}.png")   
                    else:
                        stframe1.image(eval(frame1))      
                        stframe2.image(eval(frame2))      
                
                elif len (optionOutput) == 1 : 
                    i1 = optionOutput[0].index(" ")
                    frame1 = optionOutput[0][i1+1:]
                    if method == "P3MNet":
                        stframe1.image(f"{frame1}.png")   
                    else:
                        stframe1.image(eval(frame1))      
                else :
                    if method == "P3MNet":
                        stframe1.image("matting.png")   
                    else:
                        stframe1.image(matting)
                    
                print ( "done1" )
     
     
    elif option == "Evaluation": 
        filePath = st.sidebar.selectbox('Which is dataset?',
                                      ('Test dataset', 'Custom dataset'))
        batchSize = st.sidebar.slider('Batch Size?', 0, 500, 25)



        p="/home/pcwork/ai/ftech/finger/handGestureWithMatting/evaluation/archive/Achieve"

        if filePath == 'Test dataset': 
            p="/home/pcwork/ai/ftech/finger/handGestureWithMatting/evaluation/archive/Achieve/image"
            pathAlpha = "/home/pcwork/ai/ftech/finger/handGestureWithMatting/evaluation/archive/Achieve/alpha"
            path = os.listdir(p)
            path.sort() 
        elif filePath == 'Custom dataset': 
            p= '/home/pcwork/ai/ftech/finger/handGestureWithMatting/evaluation/archive/Achieve/imageTest2' 
            pathAlpha = '/home/pcwork/ai/ftech/finger/handGestureWithMatting/evaluation/archive/Achieve/alphaTest2' 
            path = os.listdir(p) 
            path.sort() 


        if st.sidebar.button("Run"):
            model = chooseMethod(method,imgBackground)
            count = 0 
            sad = 0  
            mse = 0 
            mad = 0
            for i, pathImg in enumerate(path): 
                count +=1 

                image = cv2.imread(f"{p}/{pathImg}")
                alphaTruth = cv2.imread(f"{pathAlpha}/{pathImg[:-4]}.png",0)

                source = image.copy()
                alpha, matting = model.imageMatting(image)
                source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)                       
                matting = alphaTruth
                
                if method == "Mediapipe Matting" :
                  #  alphaPredict = cv2.cvtColor(alpha, cv2.COLOR_RGB2GRAY) 
                    matting = cv2.cvtColor(matting, cv2.COLOR_BGR2RGB)                       
                    cv2.imwrite('alphaMediapipe.png', alpha) 
                    alphaPredict = cv2.imread('alphaMediapipe.png',0) 

                elif method == "ModNet": 
                    alphaPredict = cv2.imread('alpha.png',0)
                else:
                    matting = cv2.cvtColor(matting, cv2.COLOR_BGR2RGB)                    
                    alphaPredict = cv2.cvtColor(alpha, cv2.COLOR_RGB2GRAY) 


                if len (optionOutput) == 3:
                    i1 = optionOutput[0].index(" ")
                    i2 = optionOutput[1].index(" ")
                    i3 = optionOutput[2].index(" ")
                    frame1 = optionOutput[0][i1+1:]
                    frame2 = optionOutput[1][i2+1:]
                    frame3 = optionOutput[2][i3+1:]
                    if method == "P3MNet":
                        stframe1.image(f"{frame1}.png")   
                        stframe2.image(f"{frame2}.png")   
                        stframe3.image(f"{frame3}.png")   
                    else:
                        stframe1.image(eval(frame1))      
                        stframe2.image(eval(frame2))      
                        stframe3.image(eval(frame3))      
                
                elif len ( optionOutput ) == 2 : 
                    i1 = optionOutput[0].index(" ")
                    i2 = optionOutput[1].index(" ")
                    frame1 = optionOutput[0][i1+1:]
                    frame2 = optionOutput[1][i2+1:]
                    if method == "P3MNet":
                        stframe1.image(f"{frame1}.png")   
                        stframe2.image(f"{frame2}.png")   
                    else:
                        stframe1.image(eval(frame1))      
                        stframe2.image(eval(frame2))      
                elif len (optionOutput) == 1 :     
                    i1 = optionOutput[0].index(" ")
                    frame1 = optionOutput[0][i1+1:]
                    if method == "P3MNet":
                        stframe1.image(f"{frame1}.png")   
                    else:
                        stframe1.image(eval(frame1))      
                else :
                    if method == "P3MNet":
                        stframe1.image("matting.png")   
                    else:
                        stframe1.image(matting)

                sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(alphaPredict/255, alphaTruth/255)
                sad += sad_diff 
                mse += mse_diff 
                mad += mad_diff 
                chart.add_rows(mad_diff)
                print (sad_diff,mse_diff, mad_diff )
                print (count)
                if count == batchSize: 
                    break 
            sadMean = sad/count 
            mseMean = mse/count 
            madMean = mad/count
            st.write(f"SAD Mean = {sadMean}, MSE Mean = {mseMean}, MAD Mean = {madMean}")
            print (f"sad Mean = {sadMean}, mse Mean = {mseMean}, mad Mean = {madMean}")
            print ( "done" )




      
      




                #if len (optionOutput) == 3:
                #    i1 = optionOutput[0].index(" ")
                #    i2 = optionOutput[1].index(" ")
                #    i3 = optionOutput[2].index(" ")
                #    frame1 = optionOutput[0][i1+1:]
                #    frame2 = optionOutput[1][i2+1:]
                #    frame3 = optionOutput[2][i3+1:]
                #    if method == "P3MNet":
                #        stframe1.image(f"{frame1}.png")          
                #        stframe2.image(f"{frame2}.png")          
                #        stframe3.image(f"{frame3}.png")          
                #    else:
                #        stframe1.image(eval(frame1))          
                #        stframe2.image(eval(frame2))          
                #        stframe3.image(eval(frame3))          


                #elif len ( optionOutput ) == 2 : 
                #    i1 = optionOutput[0].index(" ")
                #    i2 = optionOutput[1].index(" ")
                #    frame1 = optionOutput[0][i1+1:]
                #    frame2 = optionOutput[1][i2+1:]
                #    stframe1.image(eval(frame1))          
                #    stframe2.image(eval(frame2))          
                #elif len (optionOutput) == 1 : 
                #    i1 = optionOutput[0].index(" ")
                #    frame1 = optionOutput[0][i1+1:]
                #    stframe1.image(eval(frame1))          
                #else :
                #    stframe1.image(matting)
    
    
    
