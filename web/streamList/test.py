import streamlit as st
import cv2
import tempfile
optionOutput = []

#st.title('BACKGROUND MATTING')
st.markdown(f"<h1 style='text-align: center; color: blue;'>Background Matting</h1>", unsafe_allow_html=True)

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

def modNet(source, background): 
    alpha = None 
    matting = None 
    return source, alpha, matting 

def mediapipeMatting(source, background): 
    alpha = None  
    matting = None 
    return source, alpha, matting 

def P3MNet (source, background): 
    alpha = None 
    matting = None 
    return source, alpha, matting 

def GFMNet (source, background): 
    alpha = None 
    matting = None  
    return source, alpha, matting 

def chooseMethod (optionMethod, source, background): 
    if optionMethod == "Unet Segmentation": 
        return unetSegmentation(source, background) 
    elif optionMethod == "Robust Matting": 
        return robustMatting(source, background) 
    elif optionMethod == "ModNet": 
        return modNet(source, background )
    elif optionMethod == "Mediapipe Matting": 
        return mediapipeMatting(source, background) 
    elif optionMethod == "P3MNet":
        return P3MNet(source, background) 
    elif optionMethod == "GFMNet": 
        return GFMNet(source, background) 


if len(optionOutput) == 3 :            
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption(optionOutput[0])
        #st.markdown(f"<h1 style='text-align: center; color: red;'>{optionOutput[0]}</h1>", unsafe_allow_html=True)
        stframe1 = st.empty()          
    
    with col2:
        st.caption(optionOutput[1])
        stframe2 = st.empty()          
    
    with col3:
        st.caption(optionOutput[2])
        stframe3 = st.empty()          
elif len(optionOutput) == 2 :
    col1, col2 = st.columns(2) 

    with col1: 
        st.caption(optionOutput[0])
        stframe1 = st.empty() 
    with col2: 
        st.caption(optionOutput[1])
        stframe2 = st.empty()
else :                    
    try:
        st.caption(optionOutput[1])
        col1 = st.columns(1) 
        stframe1 = st.empty()
    except: 
        col1 = st.columns(1) 
        stframe1 = st.empty()




method = st.sidebar.selectbox(
     'Which method do you want to use?',
     ('Unet Segmentation','Robust Matting','ModNet','Mediapipe Matting','P3MNet','GFMNet'))



option = st.sidebar.selectbox(
     'How would you like to do?',
     ('Image Matting', 'Video Matting'))

background = st.sidebar.file_uploader("Upload background")
tbackground = tempfile.NamedTemporaryFile(delete=False) 
if background is not None:
    tbackground.write(background.read())                                 

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

                                                                              
        while cam.isOpened():                                                  
            ret, frame = cam.read()                                            
            # if frame is read correctly ret is True                          
            if not ret:                                                       
                print("Can't receive frame (stream end?). Exiting ...")       
                cam.release()                                                  
                break                                                         
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                       
            if len (optionOutput) == 3:
                stframe1.image(gray)                                               
                stframe2.image(gray)                                               
                stframe3.image(gray)
            elif len ( optionOutput ) == 2 : 
                stframe1.image(gray)
                stframe2.image(gray)
            else : 
                stframe1.image(gray)

                                                                      

elif option == "Image Matting": 
    f = st.sidebar.file_uploader("Upload image")
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    if f is not None:
        tfile.write(f.read())
        image = cv2.imread(tfile.name)
        gray = cv2.cvtColor( image , cv2.COLOR_BGR2RGB) 
        if len (optionOutput) == 3:
            stframe1.image(gray)          
            stframe2.image(gray)          
            stframe3.image(gray)
        elif len ( optionOutput ) == 2 : 
            stframe1.image(gray)
            stframe2.image(gray)
        else : 
            stframe1.image(gray)


