import tarfile
import dlib
import cv2
import imutils
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import wget


#DOWNLOAD DATASET
os.chdir("dataset")
print("Current directory:", os.getcwd())

#wget.download ('http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz') 

#compress
tarfile.open('ibug_300W_large_face_landmark_dataset.tar.gz')


