import cv2
import imutils
import time
import dlib
import numpy as np
from imutils import face_utils

Eye_predictor_path = "C:/Users/USER/Downloads/face landmark detection/wflw_98_landmarks.dat"
Eye_predictor = dlib.shape_predictor(Eye_predictor_path)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((12, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 12):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

#Membuat Fungsi Scaling
def convert_boundingbox(image, rect, scale):
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()

    w = scale*(x2-x1)
    h = scale*(y2-y1)

    # Variabel bounding box setelah diperbesar
    x1 = round(((x1+x2)/2)-(w/2))
    x2 = round(((x1+x2)/2)+(w/2))
    y1 = round(((y1+y2)/2)-(h/2))
    y2 = round(((y1+y2)/2)+(h/2))

    # Memastikan variabel berada di dalam Image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x2, image.shape[1])
    y2 = min(y2, image.shape[0])

    return(x1,y1,x2,y2)

# Menjalankan Detektor
print("[INFO] loading HOG + Linear SVM face detector...")
detector = dlib.get_frontal_face_detector()
total_time = 0

#process citra/video/open camera (0)
#vid_stream = cv2.VideoCapture(r"C:/Users/Tazkia/Downloads/PRATA/Face Detection/Vid9.mp4")
vid_stream = cv2.VideoCapture(1)
time.sleep(0.6)
acq_time = time.time()

while True:
    prev_time = time.time()
    ret, frame = vid_stream.read()
    frame = imutils.resize(frame, width=420, height=420)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Deteksi Wajah
    rects = detector(rgb, 0)

    scale = 1
    boxes = [convert_boundingbox(frame, r, scale) for r in rects]

    # loop over the bounding boxes
    for (x1, y1, x2, y2) in boxes:
        # draw the bounding box on our image
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    for rect in rects:
            start_time = time.time()
            eye_landmark = Eye_predictor(frame, rect)
            total_time = total_time + (time.time() - start_time)
            eye_landmark = face_utils.shape_to_np(eye_landmark) 
            for (x, y) in eye_landmark:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1) 
    #show fps
    fps_time = time.time() - prev_time
    print("FPS: ", 1.0 / fps_time)
    # show the output image
    cv2.imshow("Output",frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

print(total_time)

