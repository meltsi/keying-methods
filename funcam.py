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
    
    
    #seperate color chanels
    frame_r=np.zeros((width,height,1))
    frame_r[:,:,0]=frame4[:,:,0]
    frame_g=np.zeros((width,height,1))
    frame_g[:,:,0]=frame4[:,:,1]
    frame_b=np.zeros((width,height,1))
    frame_b[:,:,0]=frame4[:,:,2]
    
    #3. despill version 3: average
    despill3_g=np.zeros((width,height,1))
    despill3_g[(frame_g>((frame_r+frame_b)/2)).all(axis=2)]=1
    despill3=np.zeros((width,height,3))
    despill3[:,:,1]=despill3_g[:,:,0]
    
    #apply despill on frame
    frame4=np.where(despill3==1,((frame_r+frame_b)/2),frame4)
    
    
    
    
    
    #todo
    framecopy=np.zeros((width,height,3))
    framecopy[:,:,:]=frame4[:,:,:]
    
    
    
    
    
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
    
    
    framecopy=np.zeros((width,height,3))
    framecopy[:,:,:]=frame4[:,:,:]
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
    threshold1=60
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


def ChromaKeyVersion1b():
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
    resultimage[:,:,0]=((distances[:,:])*frame4[:,:,0])+((1-distances[:,:])*background[:,:,0])
    resultimage[:,:,1]=((distances[:,:])*frame4[:,:,1])+((1-distances[:,:])*background[:,:,1])
    resultimage[:,:,2]=((distances[:,:])*frame4[:,:,2])+((1-distances[:,:])*background[:,:,2])
    
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

















#Chroma Key 1c

#Chroma Key 2
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
    ycrcb_cr[:,:]=ycrcr[:,:,1]
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
    threshold1=90
    distances= np.where(distances<threshold1,0,1)
    resultimage[:,:,0]=((distances[:,:])*frame4[:,:,0])+((1-distances[:,:])*background[:,:,0])
    resultimage[:,:,1]=((distances[:,:])*frame4[:,:,1])+((1-distances[:,:])*background[:,:,1])
    resultimage[:,:,2]=((distances[:,:])*frame4[:,:,2])+((1-distances[:,:])*background[:,:,2])
    return(resultimage)




#Chroma Key 3
#Chroma key hsv version
def ChromaKeyVersion3():
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
    threshold1=90
    distances= np.where(distances<threshold1,0,1)
    resultimage[:,:,0]=((distances[:,:])*frame[:,:,0])+((1-distances[:,:])*background[:,:,0])
    resultimage[:,:,1]=((distances[:,:])*frame[:,:,1])+((1-distances[:,:])*background[:,:,1])
    resultimage[:,:,2]=((distances[:,:])*frame[:,:,2])+((1-distances[:,:])*background[:,:,2])
    return(resultimage)

#Color Key

#Luma Key 1

#Luma Key 2






#--------------------------------------------------------------------------------------------------------------
while(cap.isOpened()):
    try:
        ret, frame = cap.read()
        if ret == True:
            frame4 = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB) # Correct color space

            # Now: The fun stuff! ------------------------------------------
            #shape of frame
            width, height, channels = frame4.shape
    
            #make empty mask with same size as frame
            mask= np.zeros((width,height), np.uint8)

            
            #apply filter
            #frame4=ChromaKeyVersion1a(frame)
            #ChromaKeyVersion2()
            #ChromaKeyVersion3()
            #ColorKeyVersion1()
            #ColorKeyVersion2()
            #LumaKeyv3()
            frame4=ChromaKeyPlusDespill(frame4)

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
