from CV_Haifa.BowDB import train
from CV_Haifa.BowDB import test
from matplotlib import pyplot as plt
import cv2
import numpy as np

#
#
# 5)   Build recognizer – given an image determine if it contains the object or not.
# This routine will extract features from the image, determine the visual words these
# features represent, build BOW for the image and then classify the BOW using the classifier built in step 4.
#
# Name the Func / file name:   recognizer

# within file at top have variable testImageDirName which has the name of the directory
# which holds a test image or many test images.
# recognizer tests all images in that directory.
# For each image – displays the image and prints on it the name of the class found.
#
# (between image displays, ask for space bar press from user).
#

testImageDirName='test/'


def recognizer():
    categories = ['Airplanes', 'Motobike', 'Elephants']

    train()
    result = test(testImageDirName=testImageDirName)
    labels = result[0]
    images = result[1]
    plt.scatter(np.array(range(len(labels))), labels)
    plt.show()
    i = 0
    for img in images:
        plt.imshow(img)

        textstr = categories[int(labels[i])]
        props = dict(boxstyle='round', facecolor='wheat', alpha=1)
        plt.text(0, 0, textstr, fontsize=34, verticalalignment='top', bbox=props)
        i += 1
        plt.show()



recognizer()