#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import datetime, time
import sys
import cv2

# Hint: print() to stderr (fh#2) as the actual video stream goes to stdout (fh#1)!
print("funcam started ...",file=sys.stderr)

cap = cv2.VideoCapture(0) # TODO: You have to verify, if /dev/video0 is your actual webcam!
if not cap.isOpened(): # if not open already
    cap.open() #make sure, we will have data
# Adjust channel resolution & use actual
cap.set(3, 640)
cap.set(4, 480)








alternate_background= cv2.imread("pink.png")#todo 
alternate_background=cv2.cvtColor(alternate_background, cv2.COLOR_BGRA2RGB)
rval, frame = cap.read()
height,width, channels = frame.shape
#resize background to frame
alternate_background = cv2.resize(alternate_background, (width,height))



counter=2
counterprev=1 
#initialize array for saving current mean of all frames
meanimage=np.zeros((height,width,3), np.uint8)
framecopy=np.zeros((height,width,3))
#make copy of meanimage for calculations outside of rgb color space
meanimagecopy=np.zeros((height,width,3))
#initialize meanimage with first frame
meanimage=frame



#KEYING METHODS-----------------------------------------------------------------------

#Chroma Key Despill ------------------------------------------------------------------
def ChromaKeyPlusDespill(frame4):
    #array that returns the result
    resultimage= np.zeros((width,height,3), np.uint8)

    
    #convert every pixel in frame from rgb to yuv
    yuv = cv2.cvtColor(frame4, cv2.COLOR_RGB2YUV)
    
    #green RGB(0,255,0)
    colorbase=np.array([0,200,0])
    
    #convert base color to yuv
    chroma_key=cv2.cvtColor( np.uint8([[colorbase]] ), cv2.COLOR_RGB2YUV)[0][0]
    
    
    frame4=Despill1(frame4)
    
    #make copy of chroma_key for calculations outside of uint8
    chroma_key2=np.array([float(chroma_key[0]),float(chroma_key[1]),float(chroma_key[2])])

    
    #array that saves the euclidean distance between all pixels in current frame and chroma key
    distances=np.zeros((width,height))
    
    #copy u and v chanel for calculating distance outside of uint8
    yuv_u=np.zeros((width,height))
    yuv_u[:,:]=yuv[:,:,1]
    yuv_v=np.zeros((width,height))
    yuv_v[:,:]=yuv[:,:,2]
    
    #calculate distances
    distances=((chroma_key2[1]-yuv_u)**2+(chroma_key2[2]-yuv_v)**2)**(0.5)
       
    #kernel size
    pixel=5
    pixel_h=3
    
    #fill kernel     (kernel=np.array([[2,0,0,0,2],[0,0,1,0,0],[2,0,0,0,2]])/9)
    kernel=np.zeros((pixel,pixel_h))     
    kernel[0,0],kernel[pixel-1,0],kernel[pixel-1,pixel_h-1],kernel[0,pixel_h-1]=2/9,2/9,2/9,2/9
    kernel[int((pixel-1)/2),int((pixel_h-1)/2)]=1/9
    
    frame_g=np.zeros((width,height,1))
    frame_g[:,:,0]=frame4[:,:,1]
    frame_b=np.zeros((width,height,1))
    frame_b[:,:,0]=frame4[:,:,2]
    
    
    #apply kernel on distances
    distances=cv2.filter2D(distances,-1, kernel)
    
    
    #and frame_g>230 and frame_b>230
    
    #assign pixels with distances<threshold to foreground and >=threshold to foreground and make all other pixels transparent
    threshold1=95
    threshold2=350
    
    distances= np.where(((distances-threshold1)/(threshold2/threshold1))<=0,0, np.where(((distances-threshold1)/(threshold2/threshold1))>=1,1, 0))
    #distances= np.where(distances<=threshold1,0, np.where(distances>=threshold2,1, 0.5))
    
    #distances[(frame_g>150).all(axis=2)]=1
    #distances[(frame_b>150).all(axis=2)]=1
    
    
    #apply mask on frame
    resultimage[:,:,0]=((distances[:,:])*frame4[:,:,0])+((1-distances[:,:])*alternate_background[:,:,0])
    resultimage[:,:,1]=((distances[:,:])*frame4[:,:,1])+((1-distances[:,:])*alternate_background[:,:,1])
    resultimage[:,:,2]=((distances[:,:])*frame4[:,:,2])+((1-distances[:,:])*alternate_background[:,:,2])

    return(resultimage)

def Despill1(frame4):
    #seperate color chanels
    frame_r=np.zeros((width,height,1))
    frame_r[:,:,0]=frame4[:,:,0]
    frame_g=np.zeros((width,height,1))
    frame_g[:,:,0]=frame4[:,:,1]
    frame_b=np.zeros((width,height,1))
    frame_b[:,:,0]=frame4[:,:,2]
    
    despill_g=np.zeros((width,height,1))
    despill_g[(frame_g>frame_r).all(axis=2)]=1
    despill=np.zeros((width,height,3))
    despill[:,:,1]=despill_g[:,:,0]
    frame4=np.where(despill==1,frame_r,frame4)
    return(frame4)

def Despill2(frame4):
    #seperate color chanels
    frame_r=np.zeros((width,height,1))
    frame_r[:,:,0]=frame4[:,:,0]
    frame_g=np.zeros((width,height,1))
    frame_g[:,:,0]=frame4[:,:,1]
    frame_b=np.zeros((width,height,1))
    frame_b[:,:,0]=frame4[:,:,2]
    
    despill_g=np.zeros((width,height,1))
    despill_g[(frame_g>frame_b).all(axis=2)]=1
    despill=np.zeros((width,height,3))
    despill[:,:,1]=despill_g[:,:,0]
    frame4=np.where(despill==1,frame_b,frame4)
    return(frame4)
    
    
def Despill3(frame4):
    #seperate color chanels
    frame_r=np.zeros((width,height,1))
    frame_r[:,:,0]=frame4[:,:,0]
    frame_g=np.zeros((width,height,1))
    frame_g[:,:,0]=frame4[:,:,1]
    frame_b=np.zeros((width,height,1))
    frame_b[:,:,0]=frame4[:,:,2]
    
    #3. despill version 3: average
    despill=np.zeros((width,height,1))
    despill_g[(frame_g>((frame_r+frame_b)/2)).all(axis=2)]=1
    despill=np.zeros((width,height,3))
    despill[:,:,1]=despill_g[:,:,0]
    #apply despill on frame
    frame4=np.where(despill==1,((frame_r+frame_b)/2),frame4)
    return(frame4)


def Despill4(frame4):
    #seperate color chanels
    frame_r=np.zeros((width,height,1))
    frame_r[:,:,0]=frame4[:,:,0]
    frame_g=np.zeros((width,height,1))
    frame_g[:,:,0]=frame4[:,:,1]
    frame_b=np.zeros((width,height,1))
    frame_b[:,:,0]=frame4[:,:,2]
    
    despill_g=np.zeros((width,height,1))
    despill_g[(frame_g>(((2*frame_r)+frame_b)/3)).all(axis=2)]=1
    despill=np.zeros((width,height,3))
    despill[:,:,1]=despill_g[:,:,0]
    frame4=np.where(despill==1,(((2*frame_r)+frame_b)/3),frame4)
    return(frame4)

def Despill5(frame4):
    #seperate color chanels
    frame_r=np.zeros((width,height,1))
    frame_r[:,:,0]=frame4[:,:,0]
    frame_g=np.zeros((width,height,1))
    frame_g[:,:,0]=frame4[:,:,1]
    frame_b=np.zeros((width,height,1))
    frame_b[:,:,0]=frame4[:,:,2]
    
    despill_g=np.zeros((width,height,1))
    despill_g[(frame_g>((frame_r+(2*frame_b))/3)).all(axis=2)]=1
    despill=np.zeros((width,height,3))
    despill[:,:,1]=despill_g[:,:,0]
    frame4=np.where(despill==1,((frame_r+(2*frame_b))/3),frame4)
    return(frame4)

#Chroma Key 1a (yuv)----------------------------------------------------------------------------
#Chroma Key (yuv version)
def ChromaKeyVersion1a(frame4):
    #array that returns the result
    resultimage= np.zeros((width,height,3), np.uint8)

    #convert every pixel in frame from rgb to yuv
    yuv = cv2.cvtColor(frame4, cv2.COLOR_RGB2YUV)
    
    #green RGB(0,255,0)
    colorbase=np.array([0,200,0])
    
    #convert base color to yuv
    chroma_key=cv2.cvtColor( np.uint8([[colorbase]] ), cv2.COLOR_RGB2YUV)[0][0]
    
    
    
    frame4[:,:,:]=np.minimum(255,2*(np.maximum(0,frame4[:,:,:]-((1/2)*chroma_key))))
    
    
    
    #make copy of chroma_key for calculations outside of uint8
    chroma_key2=np.array([float(chroma_key[0]),float(chroma_key[1]),float(chroma_key[2])])

    
    #array that saves the euclidean distance between all pixels in current frame and chroma key
    distances=np.zeros((width,height))
    
    #copy u and v chanel for calculating distance outside of uint8
    yuv_u=np.zeros((width,height))
    yuv_u[:,:]=yuv[:,:,1]
    yuv_v=np.zeros((width,height))
    yuv_v[:,:]=yuv[:,:,2]
    
    #calculate distances
    distances=((chroma_key2[1]-yuv_u)**2+(chroma_key2[2]-yuv_v)**2)**(0.5)
       
    #kernel size
    pixel=5
    pixel_h=3
    
    #fill kernel     (kernel=np.array([[2,0,0,0,2],[0,0,1,0,0],[2,0,0,0,2]])/9)
    kernel=np.zeros((pixel,pixel_h))     
    kernel[0,0],kernel[pixel-1,0],kernel[pixel-1,pixel_h-1],kernel[0,pixel_h-1]=2/9,2/9,2/9,2/9
    kernel[int((pixel-1)/2),int((pixel_h-1)/2)]=1/9
    
    
    #apply kernel on distances
    cv2.filter2D(distances,-1, kernel)
    
   
    #assign pixels with distances<threshold to foreground and >=threshold to foreground
    threshold1=100
    threshold2=160
    distances= np.where(distances<threshold1,0,1)
    #distances= np.where(((distances-threshold1)/(threshold2/threshold1))<=0,0, np.where(((distances-threshold1)/(threshold2/threshold1))>=1,1, 3*((distances-threshold1)/(threshold2/threshold1))**2- 2*((distances-threshold1)/(threshold2/threshold1))**3))

    resultimage[:,:,0]=((distances[:,:])*frame4[:,:,0])+((1-distances[:,:])*alternate_background[:,:,0])
    resultimage[:,:,1]=((distances[:,:])*frame4[:,:,1])+((1-distances[:,:])*alternate_background[:,:,1])
    resultimage[:,:,2]=((distances[:,:])*frame4[:,:,2])+((1-distances[:,:])*alternate_background[:,:,2])
    
    return(resultimage)

#Chroma Key 1b ------------------------------------------------------------------------------------------------------------------------------------------------------------------
#todo desc

#updates counters after every loop
def counterplusone():
    global counter
    global counterprev
    counterprev+=1
    counter+=1

    
def updatemeanimage():
    global meanimagecopy
    meanimagecopy[:,:,:]=meanimage[:,:,:]#copy pixel-values for calculations outside of uint8
    framecopy[:,:,:]=frame4[:,:,:]#copy pixel-values for calculations outside of uint8
    meanimagecopy=(meanimagecopy*counterprev+framecopy)/counter #update meanimage
    meanimage[:,:,:]=meanimagecopy[:,:,:]#copy updated pixels back into meanimage
    counterplusone()#update counters


def ChromaKeyVersion1b(frame4):
    #array that returns the result
    resultimage= np.zeros((width,height,3), np.uint8)
    
    #make mask of current frame and of the mean of all previous frames
    mask1=makemask(frame4)
    mask2=makemask(meanimage)
    
    #combine masks
    mask3=mask1+mask2
    
    #distances=and(mask1,mask2)
    distances= np.where(mask3==2,1,0)
    
    #update meanimage for next loop
    updatemeanimage()
    
    #assign pixels with distances<threshold to foreground and >=threshold to foreground
    resultimage[:,:,0]=((distances[:,:])*frame4[:,:,0])+((1-distances[:,:])*alternate_background[:,:,0])
    resultimage[:,:,1]=((distances[:,:])*frame4[:,:,1])+((1-distances[:,:])*alternate_background[:,:,1])
    resultimage[:,:,2]=((distances[:,:])*frame4[:,:,2])+((1-distances[:,:])*alternate_background[:,:,2])
    
    return(resultimage)
    
    
def makemask(frame4):
    #convert every pixel in frame from rgb to yuv
    yuv = cv2.cvtColor(frame4, cv2.COLOR_RGB2YUV)
    
    #green RGB(0,255,0)
    colorbase=np.array([0,255,0])
    
    #convert base color to yuv
    chroma_key=cv2.cvtColor( np.uint8([[colorbase]] ), cv2.COLOR_RGB2YUV)[0][0]
    
    #make copy of chroma_key for calculations outside of uint8
    chroma_key2=np.array([float(chroma_key[0]),float(chroma_key[1]),float(chroma_key[2])])

    
    #array that saves the euclidean distance between all pixels in current frame and chroma key
    distances=np.zeros((width,height))
    
    #copy u and v chanel for calculating distance outside of uint8
    yuv_u=np.zeros((width,height))
    yuv_u[:,:]=yuv[:,:,1]
    yuv_v=np.zeros((width,height))
    yuv_v[:,:]=yuv[:,:,2]
    
    #calculate distances
    distances=((chroma_key2[1]-yuv_u)**2+(chroma_key2[2]-yuv_v)**2)**(0.5)
       
    #kernel size
    pixel=5
    pixel_h=3
    
    #fill kernel     (kernel=np.array([[2,0,0,0,2],[0,0,1,0,0],[2,0,0,0,2]])/9)
    kernel=np.zeros((pixel,pixel_h))     
    kernel[0,0],kernel[pixel-1,0],kernel[pixel-1,pixel_h-1],kernel[0,pixel_h-1]=2/9,2/9,2/9,2/9
    kernel[int((pixel-1)/2),int((pixel_h-1)/2)]=1/9
    
    #apply kernel on distances
    threshold1=120
    distances=cv2.filter2D(distances,-1, kernel)
    distances=np.where(distances<threshold1,0,1)
    return(distances)

















#Chroma Key 1c-----------------------------------------------------------------------------------


#Chroma Key 2-------------------------------------------------------------------------------------
#todo desc
def ChromaKeyVersion2(frame4):
    #array that returns the result
    resultimage= np.zeros((width,height,3), np.uint8)
    
    #convert every pixel in frame from rgb to ycrcb
    ycrcb = cv2.cvtColor(frame4, cv2.COLOR_RGB2YCrCb)
    
    #green RGB(0,255,0)
    colorbase=np.array([0,255,0])
    
    #convert base color to yuc
    chroma_key=cv2.cvtColor( np.uint8([[colorbase]] ), cv2.COLOR_RGB2YCrCb)[0][0]
    
    #make copy of chroma_key for calculations outside of uint8
    chroma_key2=np.array([float(chroma_key[0]),float(chroma_key[1]),float(chroma_key[2])])
        
    #array that saves the euclidean distance between all pixels in current frame and chroma key
    distances=np.zeros((width,height))
    
    #copy Cr and Cb chanel for calculating distance outside of uint8
    ycrcb_cr=np.zeros((width,height))
    ycrcb_cr[:,:]=ycrcb[:,:,1]
    ycrcb_cb=np.zeros((width,height))
    ycrcb_cb[:,:]=ycrcb[:,:,2]
    
    
    #calculate distances
    distances=((chroma_key2[1]-ycrcb_cr)**2+(chroma_key2[2]-ycrcb_cb)**2)**(0.5)
       
    #kernel size
    pixel=5
    pixel_h=3
    
    #fill kernel     (kernel=np.array([[2,0,0,0,2],[0,0,1,0,0],[2,0,0,0,2]])/9)
    kernel=np.zeros((pixel,pixel_h))     
    kernel[0,0],kernel[pixel-1,0],kernel[pixel-1,pixel_h-1],kernel[0,pixel_h-1]=2/9,2/9,2/9,2/9
    kernel[int((pixel-1)/2),int((pixel_h-1)/2)]=1/9

    #apply kernel on distances
    cv2.filter2D(distances,-1, kernel)
    
    #assign pixels with distances<threshold to foreground and >=threshold to foreground
    threshold1=120
    distances= np.where(distances<threshold1,0,1)
    resultimage[:,:,0]=((distances[:,:])*frame4[:,:,0])+((1-distances[:,:])*alternate_background[:,:,0])
    resultimage[:,:,1]=((distances[:,:])*frame4[:,:,1])+((1-distances[:,:])*alternate_background[:,:,1])
    resultimage[:,:,2]=((distances[:,:])*frame4[:,:,2])+((1-distances[:,:])*alternate_background[:,:,2])
    return(resultimage)




#Chroma Key 3
#todo desc
#todo fix
def ChromaKeyVersion3(frame4):
    #array that returns the result
    resultimage= np.zeros((width,height,3), np.uint8)
    
    #convert every pixel in frame from rgb to ycrcb
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    #green RGB(0,255,0)
    colorbase=np.array([0,255,0])
    
    #convert base color to yuc
    chroma_key=cv2.cvtColor( np.uint8([[colorbase]] ), cv2.COLOR_RGB2HSV)[0][0]
    
    #make copy of chroma_key for calculations outside of uint8
    chroma_key2=np.array([float(chroma_key[0]),float(chroma_key[1]),float(chroma_key[2])])
        
    #array that saves the euclidean distance between all pixels in current frame and chroma key
    distances=np.zeros((width,height))
    
    #copy Cr and Cb chanel for calculating distance outside of uint8
    hsv_s=np.zeros((width,height))
    hsv_s[:,:]=hsv[:,:,1]
    hsv_v=np.zeros((width,height))
    hsv_v[:,:]=hsv[:,:,2]
    
    
    #calculate distances
    distances=((chroma_key2[1]-hsv_s)**2+(chroma_key2[2]-hsv_v)**2)**(0.5)
       
    #kernel size
    pixel=5
    pixel_h=3
    
    #fill kernel     (kernel=np.array([[2,0,0,0,2],[0,0,1,0,0],[2,0,0,0,2]])/9)
    kernel=np.zeros((pixel,pixel_h))     
    kernel[0,0],kernel[pixel-1,0],kernel[pixel-1,pixel_h-1],kernel[0,pixel_h-1]=2/9,2/9,2/9,2/9
    kernel[int((pixel-1)/2),int((pixel_h-1)/2)]=1/9

    #apply kernel on distances
    cv2.filter2D(distances,-1, kernel)
    
    #assign pixels with distances<threshold to foreground and >=threshold to foreground
    threshold1=150
    distances= np.where(distances<threshold1,0,1)
    resultimage[:,:,0]=((distances[:,:])*frame4[:,:,0])+((1-distances[:,:])*alternate_background[:,:,0])
    resultimage[:,:,1]=((distances[:,:])*frame4[:,:,1])+((1-distances[:,:])*alternate_background[:,:,1])
    resultimage[:,:,2]=((distances[:,:])*frame4[:,:,2])+((1-distances[:,:])*alternate_background[:,:,2])
    return(resultimage)

#Color Key

def ColorKey(frame4):
    #array that returns the result
    resultimage= np.zeros((width,height,3), np.uint8)

    #green RGB(0,255,0)
    color_key=np.array([0,255,0])
    
    #make copy of chroma_key for calculations outside of uint8
    color_key2=np.array([float(color_key[0]),float(color_key[1]),float(color_key[2])])
        
    #array that saves the euclidean distance between all pixels in current frame and chroma key
    distances=np.zeros((width,height))    
    
    #
    rgb_r=np.zeros((width,height))
    rgb_r[:,:]=frame4[:,:,0]
    rgb_g=np.zeros((width,height))
    rgb_g[:,:]=frame4[:,:,1]
    rgb_b=np.zeros((width,height))
    rgb_b[:,:]=frame4[:,:,2]
    
    
    #calculate distances
    distances=((color_key2[0]-rgb_r)**2+(color_key2[1]-rgb_g)**2+(color_key2[2]-rgb_b)**2)**(0.5)
        
    #    
    threshold1=180
    distances= np.where(distances<threshold1,0,1)
    
    resultimage[:,:,0]=((distances[:,:])*frame4[:,:,0])+((1-distances[:,:])*alternate_background[:,:,0])
    resultimage[:,:,1]=((distances[:,:])*frame4[:,:,1])+((1-distances[:,:])*alternate_background[:,:,1])
    resultimage[:,:,2]=((distances[:,:])*frame4[:,:,2])+((1-distances[:,:])*alternate_background[:,:,2])
    return(resultimage)

#Luma Key 1------------------------------------------------------------------------------------

#todo add source
def smoothstep0(luminance):
    threshold1=220
    threshold2=250
    return smoothstep(luminance[0],threshold1,threshold2)

def smoothstep(x, xmin, xmax):
    smoothstep1=(x - xmin) / (xmax - xmin)
    if smoothstep1 <= 0:
        return 0
    elif smoothstep1 >= 1:
        return 1
    else:
        return 3*smoothstep1**2- 2*smoothstep1**3



    
def LumaKey(frame4):
    lumaMin=220
    lumaMax=250
    lumaMinSmooth=0
    lumaMaxSmooth=0


    #array that returns the result
    resultimage= np.zeros((width,height,3), np.uint8)

    #convert rgb value to luminance
    luma_converter=np.array([0.2989, 0.5870, 0.1140])
    luminance=np.zeros((width,height,1))
    luminance[:,:,0]=frame4[:,:,0]*luma_converter[0]+frame4[:,:,1]*luma_converter[1]+frame4[:,:,2]*luma_converter[2]
    
    #mask
    mask=np.apply_along_axis(smoothstep0,2,luminance)
  
    
    #assign resulting image to foreground or background
    resultimage[:,:,0]=(1-mask[:,:])*frame4[:,:,0]+(mask[:,:])*alternate_background[:,:,0]
    resultimage[:,:,1]=(1-mask[:,:])*frame4[:,:,1]+(mask[:,:])*alternate_background[:,:,1]
    resultimage[:,:,2]=(1-mask[:,:])*frame4[:,:,2]+(mask[:,:])*alternate_background[:,:,2]
    return(resultimage)


#Luma Key 2 ----------------------------------------------------------------------------------


def LumaKeyv2(frame4):
    threshold=130
    tolerance=10

    #array that returns the result
    resultimage= np.zeros((width,height,3), np.uint8)

    #convert every pixel in frame from rgb to yuv
    yuv = cv2.cvtColor(frame4, cv2.COLOR_RGB2YUV)
    
    luminance=np.zeros((width,height,1))
    luminance[:,:,0]=yuv[:,:,0]

    mask=np.where((0<luminance),np.where((220>luminance),1,0),1)
    
    #assign resulting image to foreground or background
    resultimage[:,:,0]=(mask[:,:,0])*frame4[:,:,0]+(1-mask[:,:,0])*alternate_background[:,:,0]
    resultimage[:,:,1]=(mask[:,:,0])*frame4[:,:,1]+(1-mask[:,:,0])*alternate_background[:,:,1]
    resultimage[:,:,2]=(mask[:,:,0])*frame4[:,:,2]+(1-mask[:,:,0])*alternate_background[:,:,2]


    return(resultimage)

#--------------------------------------------------------------------------------------------------------------






while(cap.isOpened()):
    try:
        ret, frame = cap.read()
        if ret == True:
            frame4 = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB) # Correct color space

            # Now: The fun stuff! ------------------------------------------
            #shape of frame
            width, height, channels = frame4.shape
            
            #initialize counters for number of frames (for mean-picture)   





            #make empty mask with same size as frame
            mask= np.zeros((width,height), np.uint8)

            
            #apply filter
            frame4=ChromaKeyPlusDespill(frame4)
            #frame4=ChromaKeyVersion1a(frame4)
            #frame4=ChromaKeyVersion1b(frame4)
            #frame4=ChromaKeyVersion2(frame4)
            #frame4=ChromaKeyVersion3(frame4)
            #frame4=ColorKey(frame4)
            #frame4=LumaKey(frame4)
            #frame4=LumaKeyv2(frame4)

            # End of fun stuff ... --------------------------------------------

            # Ref.: https://stackoverflow.com/questions/36579542/sending-opencv-output-to-vlc-stream
            # Write raw output (to be redirected to video device)
            sys.stdout.buffer.write(frame4.tobytes())
        else:
            print("*** Frame not read?!", file=sys.stderr)
            break
    except KeyboardInterrupt: # Quit gracefully ...
        break 
    except Exception as e:
        print("\n*** funcam aborted?!", file=sys.stderr)        
        print(str(e), file=sys.stderr)
        break

cap.release()
print("\n<<< funcam terminated. >>>\n", file=sys.stderr)
