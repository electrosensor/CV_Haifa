import cv2
import dlib
import sys
import numpy as np
import os
import time

from matplotlib import pyplot as plt

# Goal: Write an object recognition system based on Bag Of Words (BOW).

# 1)Create a DataBase –
# We will be using a set Database that can be found here.
# We will use the 3 classes of images: Airplane, Elephant and Motorbike
# They each have around 100 images each.The LAST 10 images are for testing.
#
# e) Put aside test image, (the last 10 images in the class(by name)).


class BowDB:

    n_bins = 16

    def __init__(self, dir_path):
        self.__images = []
        for img_path in os.listdir(dir_path):
            img = cv2.imread(dir_path + img_path)
            self.__images.append(img)
        print(str(len(self.__images)) + " images were loaded from " + dir_path)

    def get_train_set(self):
        length = round(len(self.__images)*0.85)
        return self.__images[:length]

    def get_test_set(self):
        length = round(len(self.__images)*0.85)
        return self.__images[length:]

    @staticmethod
    def __make_ftr_vector(image, n_feature):
        orb = cv2.ORB_create()
        desc_size = orb.descriptorSize()
        orb.setMaxFeatures(n_feature)
        keypoints = orb.detect(image)
        keypoints, desc = orb.compute(image=image, keypoints=keypoints)
        return desc, desc_size

    @staticmethod
    def __get_feature_vectors(samples, n_features=370):
        feature_vectors = []
        n_keypoints = 0
        desc_size = 0
        ranges = []
        for img in samples:
            ftr_vector, desc_size = BowDB.__make_ftr_vector(img, n_features)
            ranges.append((n_keypoints, n_keypoints + ftr_vector.shape[0]))
            n_keypoints += ftr_vector.shape[0]
            feature_vectors.append(ftr_vector)

        descriptors = np.zeros([desc_size, n_keypoints], dtype=np.uint8)
        i = 0

        for vec in feature_vectors:
            tvec = np.transpose(vec)
            descriptors[:, i:i+tvec.shape[1]] = tvec
            i += tvec.shape[1]
        return descriptors, ranges

    def get_train_ftr_vecs(self):
        return BowDB.__get_feature_vectors(self.get_train_set())

    def get_test_ftr_vecs(self):
        return BowDB.__get_feature_vectors(self.get_test_set())

    @staticmethod
    def get_clasters(feature_vectors, treshhold):
        feature_vectors = np.transpose(feature_vectors.astype(np.float32))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        retval, labels, centers = cv2.kmeans(data=feature_vectors, K=BowDB.n_bins, bestLabels=None, criteria=criteria, attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)
        #pass over D(centers,labels) and treshold it such as D(centers,labels) > treshold

        return centers, labels

    @staticmethod
    def get_frequency_hist(ftr_vectors, ranges, treshold=0):
        clasters_means, labels = BowDB.get_clasters(ftr_vectors, treshold)
        samples_hist = np.zeros([len(ranges), BowDB.n_bins], dtype=np.int32)
        distances = np.zeros(ftr_vectors.shape[1], dtype=np.int32)
        max_dist = 0
        for i in range(len(ranges)):
            relevant_lbls = labels[ranges[i][0]:ranges[i][1]]
            if treshold > 0:
                curr_ftr_vectors = ftr_vectors[:, ranges[i][0]:ranges[i][1]]
                curr_claster = clasters_means[relevant_lbls[:]]
                dist = np.linalg.norm(np.transpose(np.transpose(curr_claster[:, 0]) - curr_ftr_vectors), axis=1)

                distances[ranges[i][0]:ranges[i][1]] = dist
                relevant_lbls = relevant_lbls[dist < treshold]
                # dist = dist[dist < treshold]
                # relevant_lbls = relevant_lbls[dist > 0]

            samples_hist[i, :] = np.histogram(relevant_lbls, bins=BowDB.n_bins, range=(0, BowDB.n_bins))[0]
        if treshold > 0:
            distances = distances[distances < treshold]
            # distances = distances[distances > 0]
            max_dist = max(distances[:])
        return samples_hist, distances, max_dist

    @staticmethod
    def __false_pos(checked_class, act_labels, exp_labels):
        result = (act_labels != checked_class) & (exp_labels == checked_class)
        return sum(result)

    @staticmethod
    def __false_neg(checked_class, act_labels, exp_labels):
        result = (act_labels == checked_class) & (exp_labels != checked_class)
        return sum(result)

    @staticmethod
    def __true_pos(checked_class, act_labels, exp_labels):
        result = (act_labels == checked_class) & (exp_labels == checked_class)
        return sum(result)

    @staticmethod
    def __true_neg(checked_class, act_labels, exp_labels):
        result = (act_labels != checked_class) & (exp_labels != checked_class)
        return sum(result)

    @staticmethod
    def get_accuracy(checked_class, act_labels, exp_labels):
        TP = BowDB.__true_pos(checked_class, act_labels, exp_labels)
        TN = BowDB.__false_neg(checked_class, act_labels, exp_labels)
        accuracy = (TP + TN) / len(act_labels)
        return accuracy

    @staticmethod
    def roc_curve(checked_class, act_labels, exp_labels):
        FP = BowDB.__false_pos(checked_class, act_labels, exp_labels)
        FN = BowDB.__false_neg(checked_class, act_labels, exp_labels)
        TP = BowDB.__true_pos(checked_class, act_labels, exp_labels)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return precision, recall

    @staticmethod
    def auc(roc):
        area = np.trapz(y=roc[0], x=roc[1])
        return area

    @staticmethod
    def confusion_matrix(n_classes, act_labels, exp_labels):
        act_labels = act_labels.astype(np.int32)
        matrix = np.zeros([n_classes, n_classes], dtype=np.int32)
        for i in range(len(act_labels)):
            matrix[act_labels[i], exp_labels[i]] += 1
        return matrix


class AirplaneDB(BowDB):
    def __init__(self):
        BowDB.__init__(self, "Datasets/Airplane/")
        self.label = 0


class MotorbikeDB(BowDB):
    def __init__(self):
        BowDB.__init__(self, "Datasets/Motorbike/")
        self.label = 1


class ElephantDB(BowDB):
    def __init__(self):
        BowDB.__init__(self, "Datasets/Elephant/")
        self.label = 2


air_db = AirplaneDB()
elef_db = ElephantDB()
moto_db = MotorbikeDB()

print("Test set len of AirplaneDB is " + str(len(air_db.get_test_set())))
print("Train set len of AirplaneDB is " + str(len(air_db.get_train_set())))
print("Test set len of ElephantDB is " + str(len(elef_db.get_test_set())))
print("Train set len of ElephantDB is " + str(len(elef_db.get_train_set())))
print("Test set len of MotorbikeDB is " + str(len(moto_db.get_test_set())))
print("Train set len of MotorbikeDB is " + str(len(moto_db.get_train_set())))

# 2)   Build visual word dictionary –
# Use a feature descriptor of your choice (SIFT, HOG, SURF etc) to extract feature vectors from all your DB Images.
# Cluster the features into K clusters (use K-means, mean-shift or any clustering
# method of your choice.K is a parameter ( or window size in mean shift etc).
# This results in K visual words (usually numbered 1..K).
# Decide how you will represent each word so that later ( in Step 3 and 5)
# new features found in image will be tested against the visual words.
# Another possible system parameter will be the complexity (dimension) of the feature descriptor.

air_train_ftr_vectors, air_train_ranges = air_db.get_train_ftr_vecs()
moto_train_ftr_vectors, moto_train_ranges = moto_db.get_train_ftr_vecs()
elef_train_ftr_vectors, elef_train_ranges = elef_db.get_train_ftr_vecs()

# # Saving the objects:
# with open('weights.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(ql.weights, f)
# # Loading the objects:
# if os.path.exists('weights.pkl'):
#     with open('weights.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#         ql.weights = pickle.load(f)

# The routine created here might be run several times –
# using different parameters and different sets of images and their features.
#
# Make sure you output to a file the final dictionary you will submit to checker.And can be used
# later for testing.
#
# 3)   Create frequency histogram (BOW) for each of the images in the DB.
#      This will require writing a routine that determines for a given feature vector,
#      which visual word it represents if at all.
#      The clusters of visual words and / or their representations will be used here.
#      Another system parameter determines when a feature is NOT any visual word
#      (usually a threshold above which the feature is too far from all clusters).
#
# Make sure the routine can read a dictionary that was saved to a file ( in step 2).


air_treshold =  350
moto_treshold = 315
elef_treshold = 600

air_train_histograms, air_train_dists, air_train_max_dist = AirplaneDB.get_frequency_hist(air_train_ftr_vectors, air_train_ranges, air_treshold)
moto_train_histograms, moto_train_dists, moto_train_max_dist = MotorbikeDB.get_frequency_hist(moto_train_ftr_vectors, moto_train_ranges, moto_treshold)
elef_train_histograms, elef_train_dists, elef_train_max_dist = ElephantDB.get_frequency_hist(elef_train_ftr_vectors, elef_train_ranges, elef_treshold)

air_train_X = 0.1*air_db.label*np.ones((len(air_train_dists)))
moto_train_X = 0.1*moto_db.label*np.ones((len(moto_train_dists)))
elef_train_X = 0.1*elef_db.label*np.ones((len(elef_train_dists)))


plt.scatter(y=air_train_X , x=air_train_dists)
plt.scatter(y=moto_train_X , x=moto_train_dists)
plt.scatter(y=elef_train_X , x=elef_train_dists)
plt.legend()
plt.show()

# 4)   Given the histograms (BOWs) of the DB image –
#      build a classifier to distinguish between object and non-object.
#      Build Classifier of your choice (SVM (linear or kernel), Fisher, Cascade etc).
#
# Make sure to save the classifier and / or its parameters to external file to be used later.

# # in SVM::train @param trainData - training data that can be loaded from file using TrainData::loadFromCSV or .created with TrainData::create.
# air_train_histogramssvm = cv2.ml_SVM()
air_train_data = np.float32(air_train_histograms).reshape(-1, BowDB.n_bins)
moto_train_data = np.float32(moto_train_histograms).reshape(-1, BowDB.n_bins)
elef_train_data = np.float32(elef_train_histograms).reshape(-1, BowDB.n_bins)

train_set = np.concatenate((air_train_data, moto_train_data), axis=0)
train_set = np.concatenate((train_set, elef_train_data), axis=0)


air_responses = air_db.label*np.ones([air_train_histograms.shape[0], 1], dtype=np.int32)
moto_responses = moto_db.label*np.ones([moto_train_histograms.shape[0], 1], dtype=np.int32)
elef_responses = elef_db.label*np.ones([elef_train_histograms.shape[0], 1], dtype=np.int32)

responses = np.concatenate((air_responses, moto_responses), axis=0)
responses = np.concatenate((responses, elef_responses), axis=0)
#
# svm2_params = dict(kernel_type=cv2.SVM_RB,
#                    svm_type=cv2.SVM_C_SVC,
#                    C=1,
#                    gamma=0.2)

svm = cv2.ml_SVM.create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.trainAuto(samples=train_set, layout=cv2.ml.ROW_SAMPLE, responses=responses)
svm.save('svm_data.dat')

# 5)   Build recognizer – given an image determine if it contains the object or not.
# This routine will extract features from the image, determine the visual words these
# features represent, build BOW for the image and then classify the BOW using the classifier built in step 4.
#
# Name the Func / file name:   recognizer



air_test_ftr_vectors, air_test_ranges = air_db.get_test_ftr_vecs()
moto_test_ftr_vectors, moto_test_ranges = moto_db.get_test_ftr_vecs()
elef_test_ftr_vectors, elef_test_ranges = elef_db.get_test_ftr_vecs()

air_test_treshold =  556
moto_test_treshold = 520
elef_test_treshold = 567


air_test_histograms, air_test_dists, air_test_max_dist = AirplaneDB.get_frequency_hist(air_test_ftr_vectors, air_test_ranges, air_treshold)
moto_test_histograms, moto_test_dists, moto_test_max_dist = MotorbikeDB.get_frequency_hist(moto_test_ftr_vectors, moto_test_ranges, moto_treshold)
elef_test_histograms, elef_test_dists, elef_test_max_dist = ElephantDB.get_frequency_hist(elef_test_ftr_vectors, elef_test_ranges, elef_treshold)

air_test_X = 0.1*air_db.label*np.ones((len(air_test_dists)))
moto_test_X = 0.1*moto_db.label*np.ones((len(moto_test_dists)))
elef_test_X = 0.1*elef_db.label*np.ones((len(elef_test_dists)))


plt.scatter(y=air_test_X , x=air_test_dists)
plt.scatter(y=moto_test_X , x=moto_test_dists)
plt.scatter(y=elef_test_X , x=elef_test_dists)
plt.legend()
plt.show()

air_test_data = np.float32(air_test_histograms).reshape(-1, BowDB.n_bins)
moto_test_data = np.float32(moto_test_histograms).reshape(-1, BowDB.n_bins)
elef_test_data = np.float32(elef_test_histograms).reshape(-1, BowDB.n_bins)

test_set = np.concatenate((air_test_data, moto_test_data), axis=0)
test_set = np.concatenate((test_set, elef_test_data), axis=0)

air_exp_labels = air_db.label*np.ones([air_test_histograms.shape[0], 1], dtype=np.int32)
moto_exp_labels = moto_db.label*np.ones([moto_test_histograms.shape[0], 1], dtype=np.int32)
elef_exp_labels = elef_db.label*np.ones([elef_test_histograms.shape[0], 1], dtype=np.int32)

exp_labels = np.concatenate((air_exp_labels, moto_exp_labels), axis=0)
exp_labels = np.concatenate((exp_labels, elef_exp_labels), axis=0)

svm.load('svm_data.dat')
result = svm.predict(test_set)[1]
plt.imshow(result)
plt.show()

conf_mtrx = BowDB.confusion_matrix(3, result, exp_labels)
plt.imshow(conf_mtrx)
plt.colorbar()
plt.show()
# programPause = input("Press the <ENTER> key to continue...")

# within file at top have variable testImageDirName which has the name of the directory
# which holds a test image or many test images.
# recognizer tests all images in that directory.
# For each image – displays the image and prints on it the name of the class found.
#
# (between image displays, ask for space bar press from user).
#
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

    return 0


if __name__ == "__main__":
    main()

