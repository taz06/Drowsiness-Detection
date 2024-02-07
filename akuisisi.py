import cv2
import time

vid_stream = cv2.VideoCapture(0) #kl webcam coba (-1)    
time.sleep(0.6)

while True:
    ret, frame = vid_stream.read()


    cv2.imshow("Output", frame)
    key = cv2.waitKey(5) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break