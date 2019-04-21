import cv2
import dlib
import sys
import numpy as np
import os
import time
from CV_Haifa.BowDB import train
from CV_Haifa.BowDB import test
from CV_Haifa.BowDB import BowDB
from CV_Haifa.BowDB import AirplaneDB
from CV_Haifa.BowDB import MotorbikeDB
from CV_Haifa.BowDB import ElephantDB
from CV_Haifa.BowDB import ChairDB
from CV_Haifa.BowDB import FerryDB
from CV_Haifa.BowDB import WheelchairDB

from matplotlib import pyplot as plt

# Goal: Write an object recognition system based on Bag Of Words (BOW).

# 6)   Testing – this is the important stage of the project.
# You will report several recognition testing results.
#     Submit all the testing results in your HW report.
#     Results are quantified by Precision and Recall values(see slide 94 & 96 in lecture slides CV04 – or look online).
#     Plot in Precision - Recall ROC plot.
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

categories = ['Airplane', 'Motobike', 'Elephant']

air_precision=[]
air_recall=[]
moto_precision=[]
moto_recall=[]
elef_precision=[]
elef_recall=[]

air_acc = []
moto_acc = []
elef_acc = []

y=[]
scores=[]

min_desc_dim = 64
max_desc_dim = 256
step = 2

best_var_desc_dim=0
best_var_acc_desc_dim=0

for dd in range(min_desc_dim, max_desc_dim, step):

    print("Current descriptor dimention is: " + str(dd) + ":\n")

    train(class_list=categories, desc_n_features=dd)
    result = test(class_list=categories, desc_n_features=dd)
    act_labels = result[0]
    exp_labels = result[1]
    score = result[3]

    BowDB.get_accuracy(0, act_labels, exp_labels)

    if score > best_var_acc_desc_dim:
        best_var_desc_dim = dd
        best_var_acc_desc_dim = score
    air_roc = BowDB.roc_curve(0, act_labels, exp_labels)
    air_precision.append(air_roc[0])
    air_recall.append(air_roc[1])
    moto_roc = BowDB.roc_curve(1, act_labels, exp_labels)
    moto_precision.append(moto_roc[0])
    moto_recall.append(moto_roc[1])
    elef_roc = BowDB.roc_curve(2, act_labels, exp_labels)
    elef_precision.append(elef_roc[0])
    elef_recall.append(elef_roc[1])

    air_acc.append(BowDB.get_accuracy(0, act_labels, exp_labels))
    moto_acc.append(BowDB.get_accuracy(1, act_labels, exp_labels))
    elef_acc.append(BowDB.get_accuracy(2, act_labels, exp_labels))

    y.append(exp_labels)
    scores.append(act_labels)
    # print(air_acc)
    # print(moto_acc)
    # print(elef_acc)

#i) Plot ROC curve

plt.plot(air_precision, air_recall)
plt.plot(moto_precision, moto_recall)
plt.plot(elef_precision, elef_recall)
plt.show()

# ii) Plot Accuracy as a function of the dependent variable.

plt.plot(np.array(range(len(air_acc))), air_acc)
plt.plot(np.array(range(len(moto_acc))), moto_acc)
plt.plot(np.array(range(len(elef_acc))), elef_acc)
plt.show()

#  iii) Print best variable value according to ROC curve.
print()
print("best_variable - Descriptor dimension: Value = " + str(best_var_desc_dim) + " Accuracy =" + str(best_var_acc_desc_dim))
print()

#     b) Show the change in performance over all the data, as a function of the size of the Dictionary.
#        (Use the best threshold value found in a) ).
#        i) Plot ROC curve
#        ii) Plot Accuracy as a function of Dictionary size.
#        Accuracy is defined as: (TP + TN) /  # AllData


air_precision=[]
air_recall=[]
moto_precision=[]
moto_recall=[]
elef_precision=[]
elef_recall=[]
air_acc = []
moto_acc = []
elef_acc = []

min_dictionary_size = 4
max_dictionary_size = 128
step = 1

best_var_dict_size = 0
best_var_acc_dict_size = 0

for ds in range(min_dictionary_size, max_dictionary_size, step):
    print()
    print("Current dictionary size is: " + str(ds) + ":\n")

    train(class_list=categories, dict_size=ds, desc_n_features=best_var_desc_dim)
    result = test(class_list=categories, dict_size=ds)
    act_labels = result[0]
    exp_labels = result[1]
    score = result[3]

    if score > best_var_acc_dict_size :
        best_var_dict_size = ds
        best_var_acc_dict_size = score

    air_roc = BowDB.roc_curve(0, act_labels, exp_labels)
    air_precision.append(air_roc[0])
    air_recall.append(air_roc[1])
    moto_roc = BowDB.roc_curve(1, act_labels, exp_labels)
    moto_precision.append(moto_roc[0])
    moto_recall.append(moto_roc[1])
    elef_roc = BowDB.roc_curve(2, act_labels, exp_labels)
    elef_precision.append(elef_roc[0])
    elef_recall.append(elef_roc[1])

    air_acc.append(BowDB.get_accuracy(0, act_labels, exp_labels))
    moto_acc.append(BowDB.get_accuracy(1, act_labels, exp_labels))
    elef_acc.append(BowDB.get_accuracy(2, act_labels, exp_labels))
#i) Plot ROC curve

plt.plot(air_precision, air_recall)
plt.plot(moto_precision, moto_recall)
plt.plot(elef_precision, elef_recall)
plt.show()

# ii) Plot Accuracy as a function of the dependent variable.

plt.plot(np.array(range(len(air_acc))), air_acc)
plt.plot(np.array(range(len(moto_acc))), moto_acc)
plt.plot(np.array(range(len(elef_acc))), elef_acc)
plt.show()

print("best_variable - Dictionary Size: Value = " + str(best_var_dict_size) + " Accuracy =" + str(best_var_acc_dict_size))

###
# best_var_dict_size = 24
# best_var_desc_dim = 68
###

# c) Report performance on test images.
#
#
# Report Precision, Recall and Accuracy for:

train(class_list=categories, dict_size=best_var_dict_size, desc_n_features=best_var_desc_dim)

result = test(class_list=categories, dict_size=best_var_dict_size, desc_n_features=best_var_desc_dim)
act_labels = result[0]
exp_labels = result[1]
score = result[3]

air_roc = BowDB.roc_curve(0, act_labels, exp_labels)
air_precision = air_roc[0]
air_recall = air_roc[1]
moto_roc = BowDB.roc_curve(1, act_labels, exp_labels)
moto_precision = moto_roc[0]
moto_recall = moto_roc[1]
elef_roc = BowDB.roc_curve(2, act_labels, exp_labels)
elef_precision = elef_roc[0]
elef_recall = elef_roc[1]
# i) ALL test images

air_size = len(exp_labels[exp_labels == 0])
moto_size = len(exp_labels[exp_labels == 1])
elef_size = len(exp_labels[exp_labels == 2])
all_size = len(exp_labels)

avg_precision = air_precision*(air_size/all_size) + moto_precision*(moto_size/all_size) + elef_precision*(elef_size/all_size)
avg_recall = air_recall*(air_size/all_size) + moto_recall*(moto_size/all_size) + elef_recall*(elef_size/all_size)

print("All test best scores: Accuracy: " + str(score) + " Precision = " + str(avg_precision) + ", Recall = " + str(avg_recall))
#        ii) For each class(test images of each class ).

air_acc = result[2][0]
moto_acc = result[2][1]
elef_acc = result[2][2]

print("Aiplanes scores: Accuracy: " + str(air_acc) + " Precision = " + str(air_precision) + ", Recall = " + str(air_recall))
print("Motobikes scores: Accuracy: " + str(moto_acc) + " Precision = " + str(moto_precision) + ", Recall = " + str(moto_recall))
print("Elefants scores: Accuracy: " + str(elef_acc) + " Precision = " + str(elef_precision) + ", Recall = " + str(elef_recall))

#        iii) Plot the confusion matrix (see slide 86).

conf_matrix = BowDB.confusion_matrix(3, act_labels, exp_labels)
plt.imshow(conf_matrix)
plt.colorbar()
plt.show()


#     d) Show example images that were falsely determined as object (False Positive / False Alarm) and
#        images that were incorrectly classified as NON-object (false negatives / miss).

air_db = AirplaneDB()
moto_db = MotorbikeDB()
elef_db = ElephantDB()
air_test_set = air_db.get_test_set()
moto_test_set = moto_db.get_test_set()
elef_test_set = elef_db.get_test_set()
test_set = air_test_set + moto_test_set + elef_test_set

air_FP=[]
air_FN=[]
moto_FP=[]
moto_FN=[]
elef_FP=[]
elef_FN=[]
i=0
for i in range(0, len(act_labels)):
    if (act_labels[i] == moto_db.label or act_labels[i] == elef_db.label) and (exp_labels[i] == air_db.label):
        air_FN.append(i)
    elif (act_labels[i] == air_db.label) and (exp_labels[i] == moto_db.label or exp_labels[i] == elef_db.label):
        air_FP.append(i)
    if (act_labels[i] == air_db.label or act_labels[i] == elef_db.label) and (exp_labels[i] == moto_db.label):
        moto_FN.append(i)
    elif (act_labels[i] == moto_db.label) and (exp_labels[i] == air_db.label or exp_labels[i] == elef_db.label):
        moto_FP.append(i)
    if (act_labels[i] == air_db.label or act_labels[i] == moto_db.label) and (exp_labels[i] == elef_db.label):
        elef_FN.append(i)
    elif (act_labels[i] == elef_db.label) and (exp_labels[i] == air_db.label or exp_labels[i] == moto_db.label):
        elef_FP.append(i)


print("air_FP" + str(air_FP) + "\n")
print("air_FN" + str(air_FN) + "\n")
print("moto_FP" + str(moto_FP) + "\n")
print("moto_FN" + str(moto_FN) + "\n")
print("elef_FP" + str(elef_FP) + "\n")
print("elef_FN" + str(elef_FN) + "\n")



def fp_fn(samples, n_im, FP_FN, act_labels, type_str):
    print(type_str)
    i = 0
    cv2.namedWindow('recognizer', cv2.WINDOW_AUTOSIZE)
    for fp_fn_idx in FP_FN[:n_im]:
            while True:
                textstr = categories[int(act_labels[fp_fn_idx])]
                cv2.putText(samples[fp_fn_idx],
                            text=textstr,
                            org=(50, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 255, 0),
                            thickness=2,
                            lineType=1)
                cv2.imshow('recognizer', samples[fp_fn_idx])
                k = cv2.waitKey(20)
                if k == 32:  # space bar
                    break
            i += 1
    cv2.destroyAllWindows()

num_im=1
print("Images that were falsely determined as object (false negative or false positive) ")
fp_fn(test_set, num_im, air_FN, act_labels, 'Airplane FN: ')
fp_fn(test_set, num_im, air_FP, act_labels, 'Airplane FP: ')
fp_fn(test_set, num_im, moto_FN, act_labels, 'Motobike FN: ')
fp_fn(test_set, num_im, moto_FP, act_labels, 'Motobike FP: ')
fp_fn(test_set, num_im, elef_FN, act_labels, 'Elephant FN: ')
fp_fn(test_set, num_im, elef_FP, act_labels, 'Elephant FP: ')


#     e) Think why and where the failures of your system is and improve your object recognition system
#       (add object images, non-object images or change methods of feature descriptors / clustering / classifications ).
#        Report again (steps 6a-c) the improved results.
#        Explain / discuss in the report where the improvement occurred and why.



#     f) What happens to accuracy of your system when more classes are added?
#        The Dataset contains 3 additional classes: Chair, Wheelchair, Ferry.
#        Adding the classes one by one,
#        plot the accuracy resulting from testing on the same test data as in Step 6 c
#        testing(the test data in the 3 classes: Airplane, Elephant and Motorbike).

# Report performance on test images.
#
# Report Precision, Recall and Accuracy for:

categories = ['Airplane', 'Motobike', 'Elephant', 'Chair', 'Ferry', 'Wheelchair']

train(class_list=categories, dict_size=best_var_dict_size, desc_n_features=best_var_desc_dim)

chair_db = ChairDB()
ferry_db = FerryDB()
wheelch_db = WheelchairDB()

result = test(class_list=categories, dict_size=best_var_dict_size, desc_n_features=best_var_desc_dim)
act_labels = result[0]
exp_labels = result[1]
score = result[3]

air_roc = BowDB.roc_curve(0, act_labels, exp_labels)
air_precision = air_roc[0]
air_recall = air_roc[1]
moto_roc = BowDB.roc_curve(1, act_labels, exp_labels)
moto_precision = moto_roc[0]
moto_recall = moto_roc[1]
elef_roc = BowDB.roc_curve(2, act_labels, exp_labels)
elef_precision = elef_roc[0]
elef_recall = elef_roc[1]
chair_roc = BowDB.roc_curve(3, act_labels, exp_labels)
chair_precision = elef_roc[0]
chair_recall = elef_roc[1]
ferry_roc = BowDB.roc_curve(4, act_labels, exp_labels)
ferry_precision = elef_roc[0]
ferry_recall = elef_roc[1]
wheelch_roc = BowDB.roc_curve(5, act_labels, exp_labels)
wheelch_precision = elef_roc[0]
wheelch_recall = elef_roc[1]


# i) ALL test images

air_size = len(exp_labels[exp_labels == air_db.label])
moto_size = len(exp_labels[exp_labels == air_db.label])
elef_size = len(exp_labels[exp_labels == air_db.label])
chair_size = len(exp_labels[exp_labels == air_db.label])
ferry_size = len(exp_labels[exp_labels == air_db.label])
wheelch_size = len(exp_labels[exp_labels == wheelch_db.label])


all_size = len(exp_labels)

avg_precision = air_precision*(air_size/all_size) + \
                moto_precision*(moto_size/all_size) + \
                elef_precision*(elef_size/all_size) + \
                chair_precision*(chair_size/all_size) + \
                ferry_precision*(ferry_size/all_size) + \
                wheelch_precision*(wheelch_size/all_size)
avg_recall = air_recall*(air_size/all_size) + \
             moto_recall * (moto_size / all_size) + \
             elef_recall * (elef_size / all_size) + \
             chair_recall * (chair_size / all_size) + \
             ferry_recall * (ferry_size / all_size) + \
             wheelch_recall * (wheelch_size / all_size)

print("All test best scores: Accuracy: " + str(score) + " Precision = " + str(avg_precision) + ", Recall = " + str(avg_recall))
#        ii) For each class(test images of each class ).

air_acc = result[2][0]
moto_acc = result[2][1]
elef_acc = result[2][2]
chair_acc = result[2][3]
ferry_acc = result[2][4]
wheelch_acc = result[2][5]

print(air_db.name + " scores: Accuracy: " + str(air_acc) + '\n' +
                                " Precision = " + str(air_precision) +
                                ", Recall = " + str(air_recall))
print(moto_db.name + " scores: Accuracy: " + str(moto_acc) + '\n' +
                                " Precision = " + str(moto_precision) +
                                ", Recall = " + str(moto_recall))
print(elef_db.name + " scores: Accuracy: " + str(elef_acc) + '\n' +
                                " Precision = " + str(elef_precision) +
                                ", Recall = " + str(elef_recall))
print(chair_db.name + " scores: Accuracy: " + str(chair_acc) + '\n' +
                                " Precision = " + str(chair_precision) +
                                ", Recall = " + str(chair_recall))
print(ferry_db.name + " scores: Accuracy: " + str(ferry_acc) + '\n' +
                                " Precision = " + str(ferry_precision) +
                                ", Recall = " + str(ferry_recall))
print(wheelch_db.name + " scores: Accuracy: " + str(wheelch_acc) + '\n' +
                                " Precision = " + str(wheelch_precision) +
                                ", Recall = " + str(wheelch_recall))

#        iii) Plot the confusion matrix (see slide 86).

conf_matrix = BowDB.confusion_matrix(len(categories), act_labels, exp_labels)
plt.imshow(conf_matrix)
plt.colorbar()
plt.show()
