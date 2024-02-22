import cv2
import imutils
import time
import dlib
import numpy as np
from imutils import face_utils

Eye_predictor_path = 'shape_predictor_68_face_landmarks.dat' #ngambil data predictornya
Eye_predictor = dlib.shape_predictor(Eye_predictor_path) #memperoleh koordinat landmark

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((12, 2), dtype=dtype) #12 titik di ruang 2 dimensi
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 12):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

#Membuat Fungsi Scaling
def yunet_boundingbox(image, box):
    x1 = box[0]
    y1 = box[1]
    w1 = box[2]
    h1 = box[3]
    y_Nose = box[9]

    new_h = round(w1)
    new_w = round(w1)

    # Variabel bounding box setelah diperbesar
    bbox_x1 = round((x1+box[2]/2)-(w1/2))
    bbox_x2 = round((x1+box[2]/2)+(w1/2))
    bbox_y1 = round((y1+box[3]/2)-(h1/2))
    bbox_y2 = round((y1+box[3]/2)+(h1/2))

    '''bbox_x1 = round(x1)
    bbox_x2 = round(x1+new_w)
    bbox_y1 = round(y1)
    bbox_y2 = round(y1+new_h)'''

    # Memastikan variabel berada di dalam Image
    bbox_x1 = max(0, bbox_x1)
    bbox_y1 = max(0, bbox_y1)
    bbox_x2 = min(bbox_x2, image.shape[1])
    bbox_y2 = min(bbox_y2, image.shape[0])

    return(bbox_x1, bbox_y1, bbox_x2, bbox_y2)

#OpenCV Yunet
#Memasukkan Path Model DNN
yunet = cv2.FaceDetectorYN.create(
    model= 'face_detection_yunet_2022mar.onnx',
    config='',
    input_size=(320, 320),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

total_time = 0

#process citra/video/open camera (0)
#vid_stream = cv2.VideoCapture(r"C:/Users/Tazkia/Downloads/PRATA/Face Detection/cobanyetirr.mp4")
vid_stream = cv2.VideoCapture(0) #kl webcam (1)
time.sleep(0.6)

while True:
    prev_time = time.time()
    ret, frame = vid_stream.read()
    if frame is not None:
        frame = imutils.resize(frame, width=420, height=420)
        h, w = frame.shape[:2]

        yunet.setInputSize([w,h])
        results = yunet.detect(frame)[1]  
        #print(results[0][12])

        #Menempatkan Bounding Box
        if results is not None:
            for result in results:
                box = result[0:10].astype(np.int32)
                bbox = yunet_boundingbox(frame, box)
                (x3, y3, x4, y4) = bbox
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 1)

                #cv2.circle(frame, (result[0][0], result[0][1]), 1, (0, 0, 255), -1)

                #mengconvert bounding box menjadi dlib rectangle
                dlibrect = dlib.rectangle(int(x3),int(y3),int(x4),int(y4))
                #print(dlibrect)

                start_time = time.time()
                eye_landmark = Eye_predictor(frame, dlibrect)
                total_time = total_time + (time.time() - start_time)
                eye_landmark = face_utils.shape_to_np(eye_landmark) 
                #selected_landmark = eye_landmark[36:48] + eye_landmark[48:68] #+ eye_landmark[17:27] #
                for (x, y) in eye_landmark[36:48] : 
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                for (x, y) in eye_landmark[48:68] : 
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)           
                for (x, y) in eye_landmark[17:27] : 
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
        fps_time = time.time() - prev_time
        #print("FPS: ", 1.0 / fps_time)
        # show the output image        
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    else:
        break
#vid_stream.release()
#print(total_time)