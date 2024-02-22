import dlib
import cv2
import imutils
import os
import wget
import tarfile


#DOWNLOAD DATASET
os.chdir("dataset")
print("Current directory:", os.getcwd())

#wget.download('http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz')
#compress
#file_dataset = tarfile.open('ibug_300W_large_face_landmark_dataset.tar.gz')
with tarfile.open('ibug_300W_large_face_landmark_dataset.tar.gz', 'r:gz') as tar:
    # Get the names of all members (files and directories) in the tar file
    # members = tar.getnames()
    # save
    tar.extractall()

    # Print the tree structure
    #for member in members:
    #    print(member)



