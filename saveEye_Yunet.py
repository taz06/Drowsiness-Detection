import cv2
import dlib
from imutils.video import VideoStream
from imutils import face_utils
import imutils
from collections import OrderedDict
import math

# Apabila menggunakan SBC, gunakan library berikut
#import tflite_runtime.interpreter as tflite

# Apabila menggunakan pc/laptop, gunakan library berikut
import tensorflow.lite as tflite

import numpy as np
import pandas as pd
import time
from datetime import datetime

#Inisiasi Variabel
waktu = time.strftime("%Y_%m_%d-%I:%M:%p")
THR_eye_perclos = 0     #Threshold value ear untuk menghitung PERCLOS
THR_eye_close = 0       #Threshold value ear untuk menentukan kondisi mata terbuka/tertutup
nEyeClose = 0           #Jumlah frame dengan mata tertutup
counterEye = 0          #Counter frame dengan mata tertutup untuk menentukan jumlah kedipan
avgDurEyeClose = 0.0    #Durasi rata-rata setiap kali mata tertutup       
ear = 0.0               #Eye Aspect Ratio (rasio keterbukaan mata)
array_ear = []          #Array nilai EAR
avgEAR = 0.0            #Nilai EAR rata-rata per menit
stdevEAR = 0.0          #Standar deviasi EAR per menit
countFrame = 0          #Jumlah frame dengan wajah terdeteksi
perclos = 0.0           #Persentase jumlah frame dengan nilai EAR dibawah THR_eye_perclos
array_perclos = []      #Array nilai PERCLOS
dataframe_perclos = 15  #Panjang array PERCLOS yang akan digunakan sebagai input model deteksi-prediksi kantuk
closed = False          #Kondisi mata
Alarm = False           #Peringatan apabila mata tertutup
max_fps = 0.0           #FPS maksimal       
max_ear = 0             #Nilai EAR maksimal
min_ear = 1             #Nilai EAR minimal
n_alert = 0             #Sekuen alarm peringatan mata terpejam
kondisi_kantuk = 0      #Kondisi kantuk
estimasi_kantuk = 0     #Estiasi kapan kondisi kantuk mencapa 4

X_eyeVector = []        #Vektor landmark mata pada sumbu X
Y_eyeVector = []        #Vektor landmark mata pada sumbu y

dataeyeX = []           #Array data yang akan disimpan ke dalam file

#Kode/Nomor landmark dari masing-masing mata
FACIAL_MARKS_EYE_ID = OrderedDict([
	("right_eye", (0, 6)),
	("left_eye", (6, 12)),
])

#Deklarasi fungsi

#Mengambil bounding box daerah wajah
def boundingbox_OpenCV_Yunet(image, box):
    x1 = box[0]         #Kiri atas
    y1 = box[1]         #Kiri atas
    new_w = box[2]      #Lebar
    new_h = box[3]      #Tinggi

    # Variabel bounding box
    bbox_x1 = round((x1+box[2]/2)-(new_w/2))    #Kiri atas
    bbox_x2 = round((x1+box[2]/2)+(new_w/2))    #Kanan bawah
    bbox_y1 = round((y1+box[3]/2)-(new_h/2))    #Kiri atas
    bbox_y2 = round((y1+box[3]/2)+(new_h/2))    #Kanan bawah

    # Memastikan variabel berada di dalam batas Image
    bbox_x1 = max(0, bbox_x1)
    bbox_y1 = max(0, bbox_y1)
    bbox_x2 = min(bbox_x2, image.shape[1])
    bbox_y2 = min(bbox_y2, image.shape[0])

    return(bbox_x1,bbox_y1,bbox_x2,bbox_y2)

#Menghitung jarak 2 vektor
def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

#Menghitung EAR
def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])  #vertical landmark
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])  #horizontal landmark
    ear = (A + B) / (2.0 * C)           #eye aspect Ratio
    return ear

#Alarm Peringatan
def alert():
    text = "Subjek Terpejam"
    org = (75, 25)
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.5
    color = (0,0,255)  #(B, G, R)
    thickness = 2
    cv2.putText(frame, text, org, font, fontScale, color, thickness)

#Window live plot
class LivePlotEye:
    def __init__(self, w=640, h=280, yLimit=[0, 100], interval=0.01):

        self.yLimit = yLimit
        self.w = w
        self.h = h
        self.interval = interval
        self.imgPlotEye = np.zeros((self.h, self.w, 3), np.uint8)
        self.imgPlotEye[:] = 225, 225, 225

        cv2.rectangle(self.imgPlotEye, (0, 0),
                      (self.w, self.h),
                      (0, 0, 0), cv2.FILLED)
        self.xP = 0
        self.yP = 0

        self.yList = []

        self.xList = [x for x in range(0, 118)]
        self.ptime = 0

    def update(self, y, color=(0, 255, 255)):

        if time.time() - self.ptime > self.interval:
            # Refresh
            self.imgPlotEye[:] = 225, 225, 225
            # Draw Static Parts
            self.drawBackground()
            # Draw the text value real time              
            cv2.putText(self.imgPlotEye, "EAR = {:.2f}".format((ear)), (self.w - (155), 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1)
            cv2.putText(self.imgPlotEye, "Frame = {}".format((countFrame)), (self.w - (570), 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1)
            cv2.putText(self.imgPlotEye, "Eye Close = {}".format(nEyeClose), (self.w - (205), 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1)

            #data stored every menit
            cv2.putText(self.imgPlotEye, "PERCLOS (%)= {:.2f}".format(perclos), (self.w - (275), self.h - (10)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1)
            cv2.putText(self.imgPlotEye, "Drowsy lvl= {}".format(kondisi_kantuk), (self.w - (570), self.h - (10)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1)
            cv2.putText(self.imgPlotEye, "Time to drowsy= {}".format(estimasi_kantuk), (self.w - (570), self.h - (35)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1)
            
            self.yP = int(np.interp(y, self.yLimit, [self.h, 0]))
            self.yList.append(self.yP)

            if len(self.yList) == 110:
                self.yList.pop(0)

            for i in range(0, len(self.yList)):
                if i < 2:
                    pass
                else:
                    cv2.line(self.imgPlotEye, (int((self.xList[i - 1] * (self.w // 100))) - (self.w // 10),
                                            self.yList[i - 1]), (int((self.xList[i] * (self.w // 100)) - (self.w // 10)),
                                            self.yList[i]), color, 1)

            self.ptime = time.time()

        return self.imgPlotEye

    def drawBackground(self):
        # Draw Background Canvas
        cv2.rectangle(self.imgPlotEye, (0, 0), (self.w, self.h), (0, 0, 0), cv2.FILLED)

        # Draw Grid Lines
        for x in range(0, self.w, 50):
            cv2.line(self.imgPlotEye, (x, 0), (x, self.h), (50, 50, 50), 1)

        for y in range(0, self.h, 50):
            cv2.line(self.imgPlotEye, (0, y), (self.w, y), (50, 50, 50), 1)
            #  Y Label
            cv2.putText(self.imgPlotEye,"{:.2f}".format((self.yLimit[1] - ((y / 50) * ((self.yLimit[1] - self.yLimit[0]) / (self.h / 50))))),
                        (10, y), cv2.FONT_HERSHEY_PLAIN, 1, (150, 150, 150), 1)
            

#Inisialisasi
print("facial landmark predictor ...")
#Memasukkan Path Model DNN
yunet = cv2.FaceDetectorYN.create(
    model= "./face_detection_yunet_2022mar.onnx",
    config='',
    input_size=(320, 320),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_CUDA,
    target_id=cv2.dnn.DNN_TARGET_CUDA
)

#Model landmark predictor pada mata
predictor = dlib.shape_predictor("Dlib_ELP_1.dat")

#Model klasifikasi kondisi kantuk
classifier = tflite.Interpreter(model_path="Class_model_CNN2_15.tflite")
classifier.allocate_tensors()
input_classifier = classifier.get_input_details()
output_classifier = classifier.get_output_details()

#Model prediksi kantuk
drowsy_pred = tflite.Interpreter(model_path = "Pred_model_CNN1_15.tflite")
drowsy_pred.allocate_tensors()
input_drowsy_pred = drowsy_pred.get_input_details()
output_drowsy_pred = drowsy_pred.get_output_details()

(eLStart, eLEnd) = FACIAL_MARKS_EYE_ID["left_eye"]
(eRStart, eREnd) = FACIAL_MARKS_EYE_ID["right_eye"]

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_vid = cv2.VideoWriter('./DATA_Vid/vid_ta.mp4', fourcc, 18, (320,240))
graph_vid = cv2.VideoWriter('./DATA_Vid/vid_ta_grafik.mp4', fourcc, 18, (580,380))

#Akuisisi Citra dari kamera (source = 0)
print("Inisialisasi Camera")
vs = VideoStream(src=0).start()
time.sleep(0.6)

yPlotEye = LivePlotEye(w=580, h=380, yLimit=[0, 0.5], interval=0.01)

idxS = 1
frame = vs.read()
h, w = frame.shape[:2]
yunet.setInputSize([w,h])
yunet.detect(frame)[1]

t0 = time.time()

while True:
    start_time = time.time()
    
    frame = vs.read()
    frame = imutils.resize(frame,width=320, height=240)    #low for need speed fps
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          #Convert menuju gray
    h, w = frame.shape[:2]

    #Kode berikut untuk mendapatkan posisi piksel ditengah frame
    center_x = w/2
    center_y = h/2
    minimum_face_dist = 172 #sqrt(centerx^2 + centery^2) 

    #Face detection
    yunet.setInputSize([w, h])
    results = yunet.detect(frame)[1]

    if results is not None:
        countFrame += 1
        for result in (results):
            box = result[0:4].astype(np.int32)
            bbox = boundingbox_OpenCV_Yunet(frame, box)
            (x3, y3, x4, y4) = bbox     #Vektor x dan y dari bounding box daerah wajah
            
            #Code berikut berfungsi untuk memastikan hanya wajah driver saja yang 
            #diproses dengan menggunakan analogi wajah pengemudi terletak paling tengah
            face_cx = (x3+x4)/2
            face_cy = (y3+y4)/2
            face_center_dist = math.sqrt((face_cx - center_x)**2 + (face_cy - center_y)**2)
            if face_center_dist <= minimum_face_dist :
                face_x1 = x3
                face_x2 = x4
                face_y1 = y3
                face_y2 = y4
                minimum_face_dist = face_center_dist
        dlibrect = dlib.rectangle(int(face_x1),int(face_y1),int(face_x2),int(face_y2))

        #Eye landmark prediction
        shape = predictor(gray, dlibrect)
        shape = face_utils.shape_to_np(shape)

        #Mengambil koordinat kedua mata
        leftEye = shape[eLStart:eLEnd]
        rightEye = shape[eRStart:eREnd]

        #Menghitung EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = ((leftEAR + rightEAR) / 2.0)
        array_ear.append(ear)

        # Pada menit pertama, belum dilakukan perhitungan PERCLOS.
        # proses yang dilakukan adalah menghitung nilai rata-rata dan 
        # standar deviasi EAR sehingga diperoleh nilai THR_eye_perclos
        if idxS==1:
            if (time.time() - t0) >= 60:    #Apabila proses telah berjalan selama 1 menit
                n_ear_mata = len(array_ear) #Menghitung jumlah sampel EAR
                ar = np.array(array_ear)

                if max(array_ear)> max_ear :
                    max_ear = max(array_ear)
                if min(array_ear)< min_ear:
                    min_ear = min(array_ear)
                
                #Menghitung nilai rata-rata EAR
                totalEAR = 0
                for i in range(len(array_ear)):
                    totalEAR = totalEAR + array_ear[i]
                avgEAR = totalEAR / len(array_ear)

                #Menghitung standar deviasi EAR selama 1 menit
                for i in range(len(array_ear)):
                    stdevEAR = stdevEAR + ((array_ear[i]-avgEAR)**2)/len(array_ear)
                stdevEAR = math.sqrt(stdevEAR)

                THR_eye_perclos = avgEAR-stdevEAR       #Threshold untuk menghitung PERCLOS
                THR_eye_close   = (max_ear+min_ear)/2   #Threshold untuk menentukan kondisi mata tertutup atau terbuka

                #Menyimpan nilai yang telah diperoleh
                dataeyeX.append([idxS, perclos, None, avgEAR, stdevEAR, max_ear, min_ear, [], datetime.now().strftime('%H:%M:%S,%f')[:-6]])
                df = pd.DataFrame(dataeyeX, columns=['PerSave', 'PERCLOS', 'Blink', 'avgEAR', 'stdevEAR', 'max_ear', 'min_ear', 'arr_perclos', 'waktu'])

                df['avgEAR'] = df['avgEAR'].round(decimals=3)
                df['stdevEAR'] = df['stdevEAR'].round(decimals=3)
                df['max_ear'] = df['max_ear'].round(decimals=3)
                df['min_ear'] = df['min_ear'].round(decimals=3)

                print('process EYE aspect ratio and save data' + str(idxS))

                countFrame = 0
                array_ear = []
                stdevEAR = 0.0
                npstdevEAR = 0.0
                idxS += 1
                t0 = time.time()

        #Proses data pada menit kedua dan seterusnya
        else:
            if ear == None:
                ear = 0.0
            
            #Menentukan kondisi mata terbuka atau tertutup, dan menghitung jumlah kedipan
            if ear <= THR_eye_close :
                if closed == False:
                    nEyeClose += 1      #Menghitung berapa kali mata dari terbuka menjadi tertutup (kedip)
                    closed = True
                counterEye += 1         #Menghitung jumlah frame berurutan dengan kondisi mata tertutup
                if counterEye >= 25:    #Peringatan apabila mata tertutup selama lebih dari 2,5 detik
                    Alarm = True
            else:
                closed = False
                counterEye = 0

            if Alarm == True:
                if n_alert < 50:        #Peringatan akan terus diberikan selama 5 detik kedepan
                    alert()
                    n_alert+= 1
                else:
                    n_alert = 0
                    Alarm = False
            
            if (time.time() - t0) >= 60: #apabila proses telah berjalan selama 1 menit
                n_ear_mata = len(array_ear) #Menghitung jumlah sampel EAR
                ar = np.array(array_ear)

                if max(array_ear)> max_ear :
                    max_ear = max(array_ear)
                if min(array_ear)< min_ear:
                    min_ear = min(array_ear)

                n_tertutup = ar[ar<THR_eye_close]
                jum_tertutup = len(n_tertutup)
                n_perclos = ar[ar<THR_eye_perclos]
                jum_perclos = len(n_perclos)
                perclos = jum_perclos/n_ear_mata
                array_perclos.append(round(perclos,3))

                if len(array_perclos)<15:
                    for i in range(15-len(array_perclos)):
                         array_perclos.append(round(perclos,3))

                if len(array_perclos)>15:
                    array_perclos=array_perclos[-15:]

                if nEyeClose == 0:
                    avgDurEyeClose = None
                else:
                    avgDurEyeClose = jum_tertutup/nEyeClose   #Durasi kedipan rata-rata. Kecepatan dalam frame

                #Menghitung nilai rata-rata EAR
                totalEAR = 0
                for i in range(len(array_ear)):
                    totalEAR = totalEAR + array_ear[i]
                avgEAR = totalEAR / len(array_ear)

                #Menghitung standar deviasi EAR selama 1 menit
                for i in range(len(array_ear)):
                    stdevEAR = stdevEAR + ((array_ear[i]-avgEAR)**2)/len(array_ear)
                stdevEAR = math.sqrt(stdevEAR)

                #Klasifikasi kondisi kantuk
                input_data = np.array(array_perclos, dtype=np.float32).reshape(1,15,1)
                classifier.set_tensor(input_classifier[0]['index'], input_data)
                classifier.invoke()
                output_data_classifier = classifier.get_tensor(output_classifier[0]['index'])
                kondisi_kantuk = np.where(output_data_classifier[0] == max(output_data_classifier[0]))[0][0]

                if (kondisi_kantuk > 1) and (kondisi_kantuk<4) :
                    drowsy_pred.set_tensor(input_drowsy_pred[0]['index'], input_data)
                    drowsy_pred.invoke()
                    output_data_prediction = drowsy_pred.get_tensor(output_drowsy_pred[0]['index'])
                    estimasi_kantuk = round(output_data_prediction[0][0])
                else:
                    estimasi_kantuk = 0

                #Menyimpan nilai yang telah diperoleh
                dataeyeX.append([idxS, perclos, nEyeClose, avgEAR, stdevEAR, max_ear, min_ear, array_perclos[:15], datetime.now().strftime('%H:%M:%S,%f')[:-6]
                                 ,kondisi_kantuk, estimasi_kantuk])
                df = pd.DataFrame(dataeyeX, columns=['PerSave', 'PERCLOS', 'Blink', 'avgEAR', 'stdevEAR', 'max_ear', 'min_ear', 'arr_perclos', 'waktu'
                                                     , 'Kondisi', 'Estimasi'])

                df['PERCLOS'] = df['PERCLOS'].round(decimals=3)
                df['avgEAR'] = df['avgEAR'].round(decimals=3)
                df['stdevEAR'] = df['stdevEAR'].round(decimals=3)
                df['max_ear'] = df['max_ear'].round(decimals=3)
                df['min_ear'] = df['min_ear'].round(decimals=3)

                print('process EYE aspect ratio and save data' + str(idxS))

                countFrame = 0
                array_ear = []
                stdevEAR = 0.0
                nEyeClose = 0
                idxS += 1
                t0 = time.time()

        for (sX, sY) in shape:
            cv2.circle(frame, (sX, sY), 2, (0, 255, 255), -1)  # Menggambar dot pada landmark mata

        out_vid.write(frame)
        cv2.imshow("Frame", frame)  # Menampilkan frame video pengemudi
        

    imgPlotEye = yPlotEye.update(ear)  # Plot grafik EAR
    graph_vid.write(imgPlotEye)
    cv2.imshow("Eye Aspect Ratio Plotter", imgPlotEye)

    T = time.time() - 0
    end_time = time.time()
    elapsed_time = end_time - start_time
    FPS = 1 / elapsed_time
    print("FPS:", FPS)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q" or "Q"): #Tombol untuk menghentikan proses
        df.to_csv('./DATA_file/tes_sidang' + '.csv', sep=';', encoding='utf-8', decimal=',',
        index=False, mode='a')
        break

out_vid.release()
graph_vid.release()
cv2.destroyAllWindows()
vs.stop()
