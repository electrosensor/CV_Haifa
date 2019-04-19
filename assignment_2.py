import cv2
import dlib
import sys
import numpy as np
import os
import time
from CV_Haifa.BowDB import train
from CV_Haifa.BowDB import test
from CV_Haifa.BowDB import MotorbikeDB
from CV_Haifa.BowDB import ElephantDB

from matplotlib import pyplot as plt

# Goal: Write an object recognition system based on Bag Of Words (BOW).

# 6)   Testing – this is the important stage of the project.
# You will report several recognition testing results.
#     Submit all the testing results in your HW report.
#     Results are quantified by Precision and Recall values(see slide 94 & 96 in lecture slides CV04 – or look online).
#     Plot in Precision - Recall
#     ROC plot.
#     When requested to report performance:
#     plot the ROC curve for the system parameter and then
#     report the best parameter and the actual Precision / Recall Values.
#     Do not forget to explain what the dependent parameter is.
#     Format should be similar to:
#
#     a) Report performance of Object Recognition on the DB images:
#        You should have a dependent variable, examples:
#        the threshold on the distance to the Cluster centers(NN, KNN),
#        threshold on the reliability of the assignment to class(SVM, Fuzzy, etc),
#        etc depends on your method of clustering.
#        i) Plot ROC curve
#        ii) Plot Accuracy as a function of the dependent variable.
#        Accuracy is defined as: (TP + TN) /  # AllData
#        iii) Print best variable value according to ROC curve.
#
#     b) Show the change in performance over all the data, as a function of the size of the Dictionary.
#        (Use the best threshold value found in a) ).
#        i) Plot ROC curve
#        ii) Plot Accuracy as a function of Dictionary size.
#        Accuracy is defined as: (TP + TN) /  # AllData
#
#     c) Report performance on test images.
#        Report Precision, Recall and Accuracy for:
#        i) ALL test images
#        ii) For each class(test images of each class ).
#        iii) Plot the confusion matrix (see slide 86).
#
#     d) Show example images that were falsely determined as object (False Positive / False Alarm) and
#        images that were incorrectly classified as NON-object (false negatives / miss).
#
#     e) Think why and where the failures of your system is and improve your object recognition system
#       (add object images, non-object images or change methods of feature descriptors / clustering / classifications ).
#        Report again (steps 6a-c) the improved results.
#        Explain / discuss in the report where the improvement occurred and why.
#
#     f) What happens to accuracy of your system when more classes are added?
#        The Dataset contains 3 additional classes: Chair, Wheelchair, Ferry.
#        Adding the classes one by one,
#        plot the accuracy resulting from testing on the same test data as in Step 6 c
#        testing(the test data in the 3 classes: Airplane, Elephant and Motorbike).


def main():

    train()
    test()
    return 0


if __name__ == "__main__":
    main()

