import cv2
import dlib
import sys
import numpy as np
import os
import time
from CV_Haifa.BowDB import train
from CV_Haifa.BowDB import test
from CV_Haifa.BowDB import BowDB
from CV_Haifa.BowDB import ElephantDB

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

min_desc_dim = 16
max_desc_dim = 256
step = 16

best_var_desc_dim=0
best_var_acc_desc_dim=0

for dd in range(min_desc_dim, max_desc_dim, step):

    print("Current descriptor dimention is: " + str(dd) + ":\n")

    train(desc_n_features=dd)
    result = test(desc_n_features=dd)
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

    # from sklearn import metrics
    #
    # air_fpr, air_tpr, air_thresholds = metrics.precision_recall_curve(y_true=exp_labels, probas_pred=proba[:, 0], pos_label=0)
    # moto_fpr, moto_tpr, moto_thresholds = metrics.precision_recall_curve(y_true=exp_labels, probas_pred=proba[:, 1], pos_label=1)
    # elef_fpr, elef_tpr, elef_thresholds = metrics.precision_recall_curve(y_true=exp_labels, probas_pred=proba[:, 2], pos_label=2)

    # plt.plot(air_fpr, air_tpr)
    # plt.plot(moto_fpr, moto_tpr)
    # plt.plot(elef_fpr, elef_tpr)
    # plt.show()

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

print("best_variable - Descriptor dimension: Value = " + str(best_var_desc_dim) + " Accuracy =" + str(best_var_acc_desc_dim))


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
max_dictionary_size = 32
step = 2

best_var_dict_size = 0
best_var_acc_dict_size = 0

for ds in range(min_dictionary_size, max_dictionary_size, step):

    print("Current dictionary size is: " + str(ds) + ":\n")

    train(dict_size=ds, desc_n_features=best_var_desc_dim)
    result = test(dict_size=ds)
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

# c) Report performance on test images.
#
#
# Report Precision, Recall and Accuracy for:
train(dict_size=best_var_dict_size, desc_n_features=best_var_desc_dim)

result = test(dict_size=best_var_dict_size, desc_n_features=best_var_desc_dim)
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


# def main():
#
#     train()
#     test()
#     return 0
#
#
# if __name__ == "__main__":
#     main()
# #
