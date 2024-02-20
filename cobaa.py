import dlib
import cv2
import imutils
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


#DOWNLOAD DATASET
os.chdir("custom-dlib-landmark")
print("Current directory:", os.getcwd())

wget http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz
#compress
#tar-xvf ibug_300W_large_face_landmark_dataset.tar.gz


