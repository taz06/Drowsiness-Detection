from tkinter import font
import cv2
import time
import numpy as np

vid_stream = cv2.VideoCapture(1)    
time.sleep(0.6)
width= int(vid_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(vid_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
prev_frame_time = 0
new_fram_time = 0

out= cv2.VideoWriter('C:/Users/USER/Documents/Github/video/cambiasa30withfps.mp4v', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))
print(width, height)
while True:
    ret, frame = vid_stream.read()
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

  
    frame = cv2.flip(frame,1)
    
    cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    
    cv2.imshow("Output", frame)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF
    
    #cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)     
    
    #print("FPS: ",fps)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
