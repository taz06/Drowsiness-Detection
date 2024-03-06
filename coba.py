import time
import numpy as np
from threading import Thread
import dlib
import cv2
from multiprocessing import Process, Pipe
from collections import OrderedDict
from imutils import face_utils
from scipy.signal import savgol_filter
import math
import sys

class WebcamStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id  # default is 0 for primary camera

        # opening video capture stream
        self.cap = cv2.VideoCapture(self.stream_id)
        if self.cap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)

        fps_input_stream = int(self.cap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.cap.read()
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.vcap stream
        self.stopped = True

        # reference to the thread for reading next available frame from input stream
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads keep running in the background while the program is executing

    # method for starting the thread for grabbing next available frame in input stream
    def start(self):
        self.stopped = False
        self.t.start()

    # method for reading next frame
    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.cap.read()

            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.cap.release()

    # method for returning latest read frame
    def read(self):
        return self.frame

    # method called to stop reading frames
    def stop(self):
        self.stopped = True

    def initframe(self):
        time.sleep(.2)
        #cam = cv2.VideoCapture(0)
        _, self.frame0 = self.cap.read()
        self.height, self.width = self.frame0.shape[:2]
        print('sized original size = ',self.width,'x',self.height)
        return self.height, self.width

class LivePlotData:
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

        self.yList0 = []

        self.xList = [x for x in range(0, 132)]
        self.ptime = 0

    def update(self, dTasignal0, color=(0, 255, 255)):

        if time.time() - self.ptime > self.interval:

            # Refresh
            self.imgPlotEye[:] = 225, 225, 225
            # Draw Static Parts
            self.drawBackground()
            # Draw the text value real time                     #ratioAvgEAR
            #cv2.putText(self.imgPlotEye, "EAR = {:.2f}".format((dTasignal)), (self.w - (190), 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
            #cv2.putText(self.imgPlotEye, "Count Frame = {}".format((idxFrame))+" frame", (self.w - (450), 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
            #cv2.putText(self.imgPlotEye, status, (self.w - (450), 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

            self.yP0 = int(np.interp(dTasignal0, self.yLimit, [self.h, 0]))

            self.yList0.append(self.yP0)

            if len(self.yList0) == 132:
                self.yList0.pop(0)


            for i in range(0, len(self.yList0)):
                if i < 2:
                    pass
                else:
                    cv2.line(self.imgPlotEye, (int((self.xList[i - 1] * (self.w // 100))) - (self.w // 10),
                                            self.yList0[i - 1]), (int((self.xList[i] * (self.w // 100)) - (self.w // 10)),
                                            self.yList0[i]), color, 1)

            self.ptime = time.time()

        return self.imgPlotEye

    def drawBackground(self):
        # Draw Background Canvas
        cv2.rectangle(self.imgPlotEye, (0, 0), (self.w, self.h), (0, 0, 0), cv2.FILLED)

        # Center Line untuk menunjukkan threshold
        #cv2.line(self.imgPlotEye, (0, self.h // 3), (self.w, self.h // 3), (150, 150, 150), 2)

        # Draw Grid Lines
        for x in range(0, self.w, 50):
            cv2.line(self.imgPlotEye, (x, 0), (x, self.h), (50, 50, 50), 1)

        for y in range(0, self.h, 50):
            cv2.line(self.imgPlotEye, (0, y), (self.w, y), (50, 50, 50), 1)
            #  Y Label
            cv2.putText(self.imgPlotEye,"{:.2f}".format((self.yLimit[1] - ((y / 50) * ((self.yLimit[1] - self.yLimit[0]) / (self.h / 50))))),
                        (10, y), cv2.FONT_HERSHEY_PLAIN, 1, (150, 150, 150), 1)

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
"""object_pts = np.array([
                        # model 3D
                        (6.825897, 6.760612, 4.402142),
                        (1.330353, 7.122144, 6.903745),
                        (-1.330353, 7.122144, 6.903745),
                        (-6.825897, 6.760612, 4.402142),
                        (5.311432, 5.485328, 3.987654),
                        (1.789930, 5.393625, 4.413414),
                        (-1.789930, 5.393625, 4.413414),
                        (-5.311432, 5.485328, 3.987654),
                        (2.005628, 1.409845, 6.165652),
                        (-2.005628, 1.409845, 6.165652),
                        (2.774015, -2.080775, 5.048531),
                        (-2.774015, -2.080775, 5.048531),   #shape [54] di model 40
                        (0.000000, -3.116408, 6.097667),    #shape[57] di model 46
                        (0.000000, -7.415691, 4.070434)     #shape[8] di model 7
                        ])"""
"""object_pts = np.array([
            # model 3D dari opencv
            (0.0, 0.0, 0.0),  # Ujung hidung
            (0.0, -330.0, -65.0),  # chin
            (-225.0, 170.0, -135.0),  # sudut kiri mata
            (225.0, 170.0, -135.0),  # sudut kanan mata
            (-150.0, -150.0, -125.0),  # Pojok bibir kiri
            (150.0, -150.0, -125.0)  # pojok bibir kanan
            ])"""

object_pts = np.array([
            # model 3D
                         [0.0, 0.0, 0.0],             # Nose tip
                         [0.0, 160.0, 0.0],           #[27]
                         [0.0, -330.0, -65.0],        # Chin
                         [-475.0, 125.0, -135.0],     #[0]
                         [475.0, 125.0, -135.0],      #[16]
                         [-400.0, -130.0, -125.0],  #
                         [400.0, -130.0, -125.0],
                         [-230.0, -250.0, -125.0],
                         [230.0, -250.0, -125.0]
            ])

def calc_area_wajah(face):
    return abs((face.left() - face.right()) * (face.bottom() - face.top()))
def savitzky_golay_filter(data, window_length, poly_order):
    filtered_data = savgol_filter(data, window_length, poly_order, mode='nearest')
    return filtered_data
def cekMatrix(R):
    """
    Checks if a matrix is a rotation matrix
    :param R: np.array matrix of 3 by 3
    :return: True or False
        Return True if a matrix is a rotation matrix, False if not
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
def rotationMatrixToEulerAngles(R):
    """
    Computes the Tait–Bryan Euler angles from a Rotation Matrix.
    Also checks if there is a gymbal lock and eventually use an alternative formula
    :param R: numpy.array
        3 x 3 Rotation matrix
    :return: (roll, pitch, yaw) tuple of float numbers
        Euler angles in radians
    """
    # Calculates Tait–Bryan Euler angles from a Rotation Matrix
    assert (cekMatrix(R))  # check if it's a Rmat

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:  # check if it's a gymbal lock situation
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])

    else:  # if in gymbal lock, use different formula for yaw, pitch roll
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])
def GerakWajah(shape):
    R_flip = np.zeros((3, 3), dtype=np.float32)
    R_flip[0, 0] = 1.0
    R_flip[1, 1] = 1.0
    R_flip[2, 2] = -1.0  # flip z axis

    """image_pts = np.array([
        #titik landmark yang dijadikan proyeksi
        (shape.part(17).x, shape.part(17).y),
        (shape.part(21).x, shape.part(21).y),
        (shape.part(22).x, shape.part(22).y),
        (shape.part(26).x, shape.part(26).y),
        (shape.part(36).x, shape.part(36).y),
        (shape.part(39).x, shape.part(39).y),
        (shape.part(42).x, shape.part(42).y),
        (shape.part(45).x, shape.part(45).y),
        (shape.part(31).x, shape.part(31).y),
        (shape.part(35).x, shape.part(35).y),
        (shape.part(48).x, shape.part(48).y),
        (shape.part(54).x, shape.part(54).y),
        (shape.part(57).x, shape.part(57).y),
        (shape.part(8).x, shape.part(8).y)], dtype="double")"""
    """image_pts = np.array([
        # titik landmark yang dijadikan proyeksi ref dari opencv
        (shape.part(30).x, shape.part(30).y), #+1 = 31
        (shape.part(8).x, shape.part(8).y),   #   = 9
        (shape.part(36).x, shape.part(36).y), #   = 37
        (shape.part(45).x, shape.part(45).y), #   = 46
        (shape.part(48).x, shape.part(48).y), #   = 49
        (shape.part(54).x, shape.part(54).y)  #   = 55
        ], dtype="double")"""
    image_pts = np.array([
        # titik landmark yang dijadikan proyeksi
        (shape.part(30).x, shape.part(30).y),  # +1 = 31
        (shape.part(27).x, shape.part(27).y),  #    = 28
        (shape.part(8).x, shape.part(8).y),    #    = 9
        (shape.part(0).x, shape.part(0).y),    #    = 1
        (shape.part(16).x, shape.part(16).y),  #    = 17
        (shape.part(3).x, shape.part(3).y),    #     = 4
        (shape.part(13).x, shape.part(13).y),  # = 14
        (shape.part(5).x, shape.part(5).y),    # = 6
        (shape.part(11).x, shape.part(11).y)   # = 12
    ], dtype="double")

    #1.
    # DLT nya Levenberg-Marquardt optimization
    status, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Globally Optimal Solution to the Perspective-n-Point Problem
    #status, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)

    #1.1 refine
    rotation_vec, translation_vec = cv2.solvePnPRefineVVS(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec)

    # calc euler angle
    #2.
    #rotation_mat = cv2.Rodrigues(rotation_vec)[0]  # Rodrigues transformation untuk Rotation Vektor ke rotation Matrik
    rotation_mat = np.matrix(cv2.Rodrigues(rotation_vec)[0])

    #3. Matrik P 3x4 input projection matrix
    #projectionmatrixP = cv2.hconcat((rotation_mat, translation_vec)) # bisa pakai numpy untuk gabungkan matrix
    #* projectionmatrixP = np.hstack((rotation_mat, translation_vec))

    #4. calc euler ada dua metode : 1.decomopose projection matrik (melibatkan vektor translasi); 2. Manual dengan calc algoritma Slabaugh (Tait–Bryan angles)
    #* euler_angle = -cv2.decomposeProjectionMatrix(projectionmatrixP)[6]
    #alternatif dengan manual calculation rotation matrik ke euler dengan algortima Gregory G. Slabaugh lebih hemat waktu krn tdk menggabungkan matriok translasi
    #yaw, pitch, roll = rotationMatrixToEulerAngles(R_flip * rotation_mat)  * 180 / np.pi
    yaw, pitch, roll = rotationMatrixToEulerAngles(R_flip * rotation_mat) * 180 / np.pi
    #R_flip *
    #* return euler_angle
    return yaw, pitch, roll  #hit manual


def takeCam(kirim4gerakwajah):
    # optimized util
    cv2.setUseOptimized(True)
    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)
        except:
            print(
                "Optimized OpenCV Gagal, cek version dan library instalation")

    ctime = 0
    ptime = 0

    webcam_stream = WebcamStream(stream_id=0)  # stream_id = 0 is for primary camera
    height, width = webcam_stream.initframe()
    #print(height,' x ',width)

    # Parameters zooming adjust, semakin kecil semakin lebar
    #zoom_w_scale_GerakWajah = 1
    #zoom_h_scale_GerakWajah = 1


    # rasio zooming GerakWajah
    #zoom_width_GerakWajah = int(width / zoom_w_scale_GerakWajah)
    #zoom_height_GerakWajah = int(height / zoom_h_scale_GerakWajah)
    # titik zooming area EAR (center frame)
    #start_x_GerakWajah = int((width - zoom_width_GerakWajah) / 2)
    #end_x_GerakWajah = start_x_GerakWajah + zoom_width_GerakWajah
    #start_y_GerakWajah = int((height - zoom_height_GerakWajah) / 2)
    #end_y_GerakWajah = start_y_GerakWajah + zoom_height_GerakWajah

    webcam_stream.start()

    #num_frames_processed = 0

    while True:

        try:

            ctime = time.perf_counter()
            fps_viewCam = 1.0 / float(ctime - ptime)
            ptime = ctime

            if webcam_stream.stopped is True:
                break
            else:
                frame0 = webcam_stream.read()
                frame = frame0
                frame = cv2.flip(frame, 2)
                frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frameGray = cv2.bilateralFilter(frameGray, 5, 10, 10)  # filter bilateral reduksi noise

                #frameGerakWajah = frameGray[start_y_GerakWajah:end_y_GerakWajah, start_x_GerakWajah:end_x_GerakWajah]

                # re-up
                up_width = 640 #  480
                up_height =  360 # 270
                up_points = (up_width, up_height)
                # resize up image
                #frameEAR = cv2.resize(frameEAR, up_points, interpolation=cv2.INTER_LINEAR)
                frameGerakWajah = cv2.resize(frameGray, up_points, interpolation=cv2.INTER_LINEAR)

                kirim4gerakwajah.send(frameGerakWajah)

            # adding a delay for simulating time taken for processing a frame
            #delay = 0.03  # delay value in seconds. so, delay=1 is equivalent to 1 second
            #time.sleep(delay)
            #num_frames_processed += 1
            frame0 = cv2.resize(frame0, (480, 270), interpolation=cv2.INTER_NEAREST)
            cv2.putText(frame0, "FPS view Original Cam = " + str(round(fps_viewCam, 0)), (5, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0, 100, 225), 2)
            cv2.imshow('frame', frame0)


            esc = cv2.waitKey(1)
            if esc == 27:
                break

        except EOFError:
            print('Communication Close')
            kirim4gerakwajah.send("END")
            kirim4gerakwajah.close()
            break


    webcam_stream.stop()


    # closing all windows
    cv2.destroyAllWindows()
    print('Closing frame di take cam')

    kirim4gerakwajah.send("END")
    kirim4gerakwajah.close()

def hitungGerakWajah(connection_obj):
    cv2.setUseOptimized(True)
    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)
        except:
            print(
                "Optimized OpenCV Gagal, cek version dan library instalation")

    yPlotX = LivePlotData(w=480, h=320, yLimit=[-10, 40], interval=0.01)
    yPlotY = LivePlotData(w=480, h=320, yLimit=[-75, 75], interval=0.01)

    dataYaw = []
    dataPitch = []

    detectorFace_GerakWajah = dlib.get_frontal_face_detector()
    predictorFullFace = dlib.shape_predictor('./predictorFace.dat')
    print('predictorFullFace')

    ctime = 0
    ptime = 0

    idxFrame_Timer = 0

    while True:
        dataFrame = connection_obj.recv()
        gray = dataFrame

        try:
            ctime = time.perf_counter()
            fps_GerakWajah = 1.0 / float(ctime - ptime)
            ptime = ctime

            if isinstance(dataFrame, str) and dataFrame == "END":
                cv2.destroyAllWindows()
                connection_obj.close()
                break

            facefullDetected = detectorFace_GerakWajah(gray, 0)

            if len(facefullDetected) > 0:
                # deteksi wajah yang paling besar untuk driver agar proses perhitungan hanya pada bounding box face driver
                facefullDetected = sorted(facefullDetected, key=calc_area_wajah, reverse=True)  # dengan menghitung luasan bounding box
                FaceFullDriver = facefullDetected[0]  # bounding box dengan area yg paling luas

                shapeCalc = predictorFullFace(gray, FaceFullDriver) # untuk di hitung
                
                shapeView = face_utils.shape_to_np(shapeCalc)  # untuk di tampilkan perlu di convert menjadi numpy array
                print(shapeView)
                #* euler_angle = get_head_pose(shape) # jika pakai decomposeProjectionMatrix dari opencv
                yaw, pitch, roll = GerakWajah(shapeCalc) # hit manual

                # buffer data
                dataYaw.append(yaw)
                dataPitch.append(pitch)

                if (dataYaw == None) & (dataPitch == None):
                    dataYaw = 0.0
                    dataPitch = 0.0
                if (len(dataYaw)) & (len(dataPitch)) > 5:
                    dataYaw.pop(0)
                    dataPitch.pop(0)

                if (len(dataYaw)) & (len(dataPitch)) >= 5:
                    filtered_data_yaw   = savitzky_golay_filter(dataYaw, 5, 2)
                    filtered_data_pitch = savitzky_golay_filter(dataPitch,5,2)

                    imgPlotX = yPlotX.update(filtered_data_yaw[-1])
                    imgPlotY = yPlotY.update(filtered_data_pitch[-1])
                    cv2.imshow("Head [Y] Angle Euler", imgPlotX)
                    cv2.imshow("Head [X] Angle Euler", imgPlotY)

                nFrames = 150
                if (idxFrame_Timer >= nFrames):  # fpsAvg * jumlah detik yang diinginkan

                    idxFrame_Timer = 0  # pereset counting frame
                    """--perhitungan disinia"""
                    #print('Tercapai di routine gerak wajah')

                idxFrame_Timer += 1


            '''cv2.putText(dataFrame, "FPS frame Gerak Wajah = " + str(round(fps_GerakWajah, 0)), (5, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0, 125, 225), 1)
            cv2.imshow("Frame Gerak Kepala", dataFrame)'''
            #cv2.imshow("Frame Gerak Kepala", dataFrame)
        except EOFError:
            print('Communication end')
            break

        cv2.waitKey(1)

    cv2.destroyAllWindows()
    print('Closing Hitung Gerakan Kepala')


if __name__ == '__main__':
    # init pipe
    terima, kirim = Pipe(duplex=False)

    prosesAkuisisiCitra = Process(target=takeCam, args=(kirim,))
    prosesEkstraksiPostural = Process(target=hitungGerakWajah, args=(terima,))

    # run processes
    prosesAkuisisiCitra.start()
    prosesEkstraksiPostural.start()

    # wait until processes finish
    prosesAkuisisiCitra.join()
    prosesEkstraksiPostural.join()
