import cv2
import time

vid_stream = cv2.VideoCapture(1) #kl webcam coba (-1)    
time.sleep(0.6)
width= int(vid_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(vid_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

out= cv2.VideoWriter('C:/Users/Tazkia/Documents/GitHub/Drowsiness-Detection/hasil/cambiasa30.mp4v', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))
print(width, height)
while True:
    ret, frame = vid_stream.read()
    frame = cv2.flip(frame,1)
    cv2.imshow("Output", frame)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

