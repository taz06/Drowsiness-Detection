import cv2
import imutils
import time
import dlib
import numpy as np
from imutils import face_utils
from collections import OrderedDict
from datetime import datetime
import pandas as pd

#x_eye = [] #vektor landmark mata sumbu x
#y_eye = [] #vektor landmark mata sumbu y

#--------TIME SETTING------
'''
format code list
https://www.programiz.com/python-programming/datetime/strftime
'''

# variabel untuk menghitung waktu stiap tahap
prep_time = 0
facedet_time = 0
lm_time = 0
convert_time = 0
datawaktu = []


# variabel frame dan fps
countframe = 0

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
    #y_Nose = box[9]

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
    
#Model Deteksi wajah OpenCV Yunet
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

#Model Landmark Dlib
#predictor_path = 'model_adv.dat' #ngambil data predictorny
lm_predictor2 = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #memperoleh koordinat landmark
lm_predictor = dlib.shape_predictor('model_adv.dat') #memperoleh koordinat landmark


total_time = 0
#process citra/video/open camera (0)
#vid_stream = cv2.VideoCapture(r"C:/Users/Tazkia/Downloads/PRATA/Face Detection/cobanyetirr.mp4")
vid_stream = cv2.VideoCapture(0) #kl webcam (1)
#time.sleep(0.6)


while True:
    ret, frame = vid_stream.read()   
    #if frame is not None:

    #pre proc
    t0 = cv2.getTickCount()
    frame = imutils.resize(frame, width=320, height=240)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]
    #print(h, w)
    cx = w/2
    cy = h/2
    #print(cx,cy)

    prep_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency() # menghitung waktu preproc
    print("preproc time : ", prep_time) #dlm second

    t1 = cv2.getTickCount()
    yunet.setInputSize([w,h])
    results = yunet.detect(frame)[1]  
    facedet_time = (cv2.getTickCount() - t1) / cv2.getTickFrequency()
    print("face detection time :", facedet_time)

    #Menempatkan Bounding Box dan landmark
    if results is not None:
        countframe += 1
        #t2 = cv2.getTickCount()
        for result in results:
            #print(result)
            box = result[0:4].astype(np.int32)
            bbox = yunet_boundingbox(frame, box)
            (x3, y3, x4, y4) = bbox

            # 1 wajah (driver saja) yang dideteksi dan proses
            # wajah driver = paling dekat dan paling tengah
            # BELUM

            #cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 165, 250), 1)

            #cv2.circle(frame, (result[0][0], result[0][1]), 1, (0, 0, 255), -1)

            #mengconvert bounding box menjadi dlib rectangle
        dlibrect = dlib.rectangle(int(x3),int(y3),int(x4),int(y4))
        #convert_time = (cv2.getTickCount() - t2) / cv2.getTickFrequency()
        #print("converting time :", convert_time)
        
        t2 = cv2.getTickCount()
        lm = lm_predictor(frame, dlibrect)
        lm_time = (cv2.getTickCount() - t2) / cv2.getTickFrequency()
        print(cv2.getTickFrequency())
        print("landmark time :", lm_time)
        #total_time = total_time + (time.time() - start_time)
        
        t3 = cv2.getTickCount()
        lm2 = lm_predictor2(frame, dlibrect)
        lm2_time = (cv2.getTickCount() - t3) / cv2.getTickFrequency()
        print("landmark 68 time :", lm2_time)

        lm = face_utils.shape_to_np(lm) 
        # Visualisasi Landmark
        for (x, y) in lm : 
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    #print("FPS: ", 1.0 / T)
    # show the output image    
    datawaktu.append([countframe, prep_time, facedet_time, lm_time, lm2_time])
    df = pd.DataFrame(datawaktu, columns = ['frame','prep time', 'face detect time', 'custom landmark time', 'landmark 68 time'])
    print("frame saat ini :", countframe)    
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        print(df)
        # untuk nyimpen hasil
        #df.to_csv('data bener setelah preprocessing' + '.csv', encoding='utf-8', decimal='.', index=False, mode='a')
        break
    #else:
    #    break
#vid_stream.release()
#print(total_time)