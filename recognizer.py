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

# within file at top have variable testImageDirName which has the name of the directory
# which holds a test image or many test images.
# recognizer tests all images in that directory.
# For each image – displays the image and prints on it the name of the class found.
#
# (between image displays, ask for space bar press from user).
#

testImageDirName='test/'


def recognizer():
    categories = ['Airplane', 'Motobike', 'Elephant']

    train()
    result = test(testImageDirName=testImageDirName)
    labels = result[0]
    images = result[1]
    plt.scatter(np.array(range(len(labels))), labels)
    plt.show()
    i = 0
    cv2.namedWindow('recognizer', cv2.WINDOW_AUTOSIZE)
    for img in images:
        while True:
            textstr = categories[int(labels[i])]
            cv2.putText(img,
                        text=textstr,
                        org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=1)
            cv2.imshow('recognizer', img)
            k = cv2.waitKey(20)
            if k == 32:  # space bar
                break
        i += 1
    cv2.destroyAllWindows()




recognizer()