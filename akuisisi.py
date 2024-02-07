import cv2
import time

vid_stream = cv2.VideoCapture(0) #kl webcam coba (-1)    
time.sleep(0.6)
width= int(vid_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(vid_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

out= cv2.VideoWriter('/hasil/baa.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, (width,height))
while True:
    ret, frame = vid_stream.read()

    cv2.imshow("Output", frame)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

