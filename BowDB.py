import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import dlib

# 1)Create a DataBase –
# We will be using a set Database that can be found here.
# We will use the 3 classes of images: Airplane, Elephant and Motorbike
# They each have around 100 images each.The LAST 10 images are for testing.
#
# e) Put aside test image, (the last 10 images in the class(by name)).

DEFAULT_TRESHOLD = 840
DEFAULT_DESC_DIM = 72
DEFAULT_DICT_SIZE = 4

class BowDB:

    n_bins = 32

    def __init__(self, dir_path):
        self.__images = []
        for img_path in os.listdir(dir_path):
            # print(str(img_path))
            img = cv2.imread(dir_path + img_path)
            self.__images.append(img)
        print(str(len(self.__images)) + " images were loaded from " + dir_path)

    @staticmethod
    def set_n_bins(k):
        BowDB.n_bins = k

    def get_train_set(self):
        length = round(len(self.__images)*0.85)
        return self.__images[:length]

    def get_test_set(self):
        length = round(len(self.__images)*0.85)
        return self.__images[length:]

    def get_all_data(self):
        return self.__images[:]

    @staticmethod
    def __make_ftr_vector(image, n_feature):
        orb = cv2.ORB_create()
        desc_size = orb.descriptorSize()
        orb.setMaxFeatures(n_feature)
        keypoints = orb.detect(image)
        keypoints, desc = orb.compute(image=image, keypoints=keypoints)
        return desc, desc_size

    @staticmethod
    def __get_feature_vectors(samples, n_features):
        feature_vectors = []
        n_keypoints = 0
        desc_size = 0
        ranges = []
        for img in samples:
            ftr_vector, desc_size = BowDB.__make_ftr_vector(img, n_features)
            ranges.append([n_keypoints, n_keypoints + ftr_vector.shape[0]])
            n_keypoints += ftr_vector.shape[0]
            feature_vectors.append(ftr_vector)

        descriptors = np.zeros([desc_size, n_keypoints], dtype=np.uint8)
        i = 0
        for vec in feature_vectors:
            tvec = np.transpose(vec)
            descriptors[:, i:i+tvec.shape[1]] = tvec
            i += tvec.shape[1]
        return descriptors, ranges

    def get_train_ftr_vecs(self, n_features=DEFAULT_DESC_DIM):
        return BowDB.__get_feature_vectors(self.get_train_set(), n_features)

    def get_test_ftr_vecs(self, n_features=DEFAULT_DESC_DIM):
        return BowDB.__get_feature_vectors(self.get_test_set(), n_features)

    def get_all_ftr_vecs(self, n_features=DEFAULT_DESC_DIM):
        return BowDB.__get_feature_vectors(self.__images, n_features)

    @staticmethod
    def get_test_clasters(feature_vectors, means=0):
        feature_vectors = np.transpose(feature_vectors.astype(np.float32))
        hist = np.zeros([1, BowDB.n_bins], dtype=np.uint32)
        for vec in feature_vectors:
            distances = []
            for i in range(len(means)):
                distances.append(np.linalg.norm(means[i]-vec))
            claster = distances.index(min(distances))
            hist[:, claster] += 1
        labels = hist
        return labels

    @staticmethod
    def get_train_clasters(feature_vectors):
        feature_vectors = np.transpose(feature_vectors.astype(np.float32))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        retval, labels, centers = cv2.kmeans(data=feature_vectors, K=BowDB.n_bins, bestLabels=None, criteria=criteria, attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)
        return centers, labels

    @staticmethod
    def get_one_test_freq_hist(ftr_vector, clasters_means=0):
        hist = BowDB.get_test_clasters(ftr_vector, means=clasters_means)
        return hist

    @staticmethod
    def get_test_freq_hist(ftr_vectors, ranges, clasters_means=0):
        samples_hist = np.zeros([len(ranges), BowDB.n_bins], dtype=np.int32)
        for i in range(len(ranges)):
            idx_from = ranges[i][0]
            idx_to = ranges[i][1]
            hist = BowDB.get_test_clasters(ftr_vectors[:, idx_from:idx_to], means=clasters_means)
            samples_hist[i, :] = hist

        return samples_hist

    @staticmethod
    def get_train_freq_hist(ftr_vectors, ranges, treshold=0):
            clasters_means, labels = BowDB.get_train_clasters(ftr_vectors)
            samples_hist = np.zeros([len(ranges), BowDB.n_bins], dtype=np.int32)
            for i in range(len(ranges)):
                idx_from = ranges[i][0]
                idx_to = ranges[i][1]
                relevant_lbls = labels[idx_from:idx_to]
                samples_hist[i, :] = np.histogram(relevant_lbls, bins=BowDB.n_bins, range=(0, BowDB.n_bins))[0]
            return samples_hist, clasters_means

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
    def get_avg_accuracy_3class(act_labels, exp_labels):
        first_acc = BowDB.get_accuracy(0, act_labels, exp_labels)
        first_size = len(exp_labels[exp_labels == 0])
        second_acc = BowDB.get_accuracy(1, act_labels, exp_labels)
        second_size = len(exp_labels[exp_labels == 1])
        third_acc = BowDB.get_accuracy(2, act_labels, exp_labels)
        third_size = len(exp_labels[exp_labels == 2])
        all_size = len(exp_labels)
        accuracy = first_acc*(first_size/all_size) + second_acc*(second_size/all_size) + third_acc*(third_size/all_size)
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


def train(treshold = DEFAULT_TRESHOLD, desc_n_features=DEFAULT_DESC_DIM, dict_size=DEFAULT_DICT_SIZE):
    air_db = AirplaneDB()
    moto_db = MotorbikeDB()
    elef_db = ElephantDB()

    print("Train set len of AirplaneDB is " + str(len(air_db.get_train_set())))
    print("Train set len of MotorbikeDB is " + str(len(moto_db.get_train_set())))
    print("Train set len of ElephantDB is " + str(len(elef_db.get_train_set())))

    # 2)   Build visual word dictionary –

    air_train_ftr_vectors, air_train_ranges = air_db.get_train_ftr_vecs(n_features=desc_n_features)
    moto_train_ftr_vectors, moto_train_ranges = moto_db.get_train_ftr_vecs(n_features=desc_n_features)
    elef_train_ftr_vectors, elef_train_ranges = elef_db.get_train_ftr_vecs(n_features=desc_n_features)

    BowDB.set_n_bins(dict_size)

    # 3)   Create frequency histogram (BOW) for each of the images in the DB.

    air_last_idx = air_train_ranges[len(air_train_ranges)-1][1]
    moto_last_idx = moto_train_ranges[len(moto_train_ranges)-1][1]
    for i in range(len(moto_train_ranges)):
        moto_train_ranges[i][0] += air_last_idx
        moto_train_ranges[i][1] += air_last_idx
    for i in range(len(elef_train_ranges)):
        elef_train_ranges[i][0] += moto_last_idx
        elef_train_ranges[i][1] += moto_last_idx

    train_ftr_vectors = np.concatenate((air_train_ftr_vectors, moto_train_ftr_vectors), axis=1)
    train_ftr_vectors = np.concatenate((train_ftr_vectors, elef_train_ftr_vectors), axis=1)

    train_ranges = air_train_ranges + moto_train_ranges + elef_train_ranges

    train_histograms, means = BowDB.get_train_freq_hist(train_ftr_vectors, train_ranges, treshold)

    # 4)   Given the histograms (BOWs) of the DB image

    train_set = np.float32(train_histograms).reshape(-1, BowDB.n_bins)

    air_responses = air_db.label * np.ones([len(air_db.get_train_set()), 1], dtype=np.int32)
    moto_responses = moto_db.label * np.ones([len(moto_db.get_train_set()), 1], dtype=np.int32)
    elef_responses = elef_db.label * np.ones([len(elef_db.get_train_set()), 1], dtype=np.int32)

    responses = np.concatenate((air_responses, moto_responses), axis=0)
    responses = np.concatenate((responses, elef_responses), axis=0)

    sc = StandardScaler()
    train_set_std = sc.fit_transform(train_set)

    svm = cv2.ml_SVM.create()
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(50.0)
    svm.setGamma(0.0006)

    svm.train(samples=train_set_std, layout=cv2.ml.ROW_SAMPLE, responses=responses)
    svm.save('svm_data.dat')

    result = svm.predict(train_set_std)[1]
    # plt.scatter(np.array(range(len(result))), result)
    # plt.show()
    #
    # conf_matrix = BowDB.confusion_matrix(3, result, responses)
    # plt.imshow(conf_matrix)
    # plt.colorbar()
    # plt.show()

 # Saving the objects:
    with open('means.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(means, f)

def test(treshold = DEFAULT_TRESHOLD, testImageDirName='', desc_n_features=DEFAULT_DESC_DIM, dict_size=DEFAULT_DICT_SIZE):

    svm = cv2.ml_SVM.create()
    svm = svm.load('svm_data.dat')
    BowDB.set_n_bins(dict_size)

    means = np.float32([BowDB.n_bins])
    # Loading the objects:
    if os.path.exists('means.pkl'):
        with open('means.pkl', 'rb') as f:
            means = pickle.load(f)

    if(testImageDirName is ''):
        air_db = AirplaneDB()
        moto_db = MotorbikeDB()
        elef_db = ElephantDB()

        print("Test set len of AirplaneDB is " + str(len(air_db.get_test_set())))
        print("Test set len of MotorbikeDB is " + str(len(moto_db.get_test_set())))
        print("Test set len of ElephantDB is " + str(len(elef_db.get_test_set())))

        air_test_ftr_vectors, air_test_ranges = air_db.get_test_ftr_vecs(n_features=desc_n_features)
        moto_test_ftr_vectors, moto_test_ranges = moto_db.get_test_ftr_vecs(n_features=desc_n_features)
        elef_test_ftr_vectors, elef_test_ranges = elef_db.get_test_ftr_vecs(n_features=desc_n_features)

        test_ftr_vectors = np.concatenate((air_test_ftr_vectors, moto_test_ftr_vectors), axis=1)
        test_ftr_vectors = np.concatenate((test_ftr_vectors, elef_test_ftr_vectors), axis=1)

        air_last_idx = air_test_ranges[len(air_test_ranges) - 1][1]
        moto_last_idx = air_last_idx + moto_test_ranges[len(moto_test_ranges) - 1][1]
        for i in range(len(moto_test_ranges)):
            moto_test_ranges[i][0] += air_last_idx
            moto_test_ranges[i][1] += air_last_idx
        for i in range(len(elef_test_ranges)):
            elef_test_ranges[i][0] += moto_last_idx
            elef_test_ranges[i][1] += moto_last_idx

        test_ranges = air_test_ranges + moto_test_ranges + elef_test_ranges

        test_histograms = BowDB.get_test_freq_hist(test_ftr_vectors, test_ranges, means)

        test_set = np.float32(test_histograms).reshape(-1, BowDB.n_bins)

        air_exp_labels = air_db.label * np.ones([len(air_db.get_test_set()), 1], dtype=np.int32)
        moto_exp_labels = moto_db.label * np.ones([len(moto_db.get_test_set()), 1], dtype=np.int32)
        elef_exp_labels = elef_db.label * np.ones([len(elef_db.get_test_set()), 1], dtype=np.int32)

        exp_labels = np.concatenate((air_exp_labels, moto_exp_labels), axis=0)
        exp_labels = np.concatenate((exp_labels, elef_exp_labels), axis=0)

        sc = StandardScaler()

        test_set_std = sc.fit_transform(test_set)

        result = svm.predict(test_set_std)[1]
        # plt.scatter(np.array(range(len(result))), result)
        # plt.show()

        # conf_matrix = BowDB.confusion_matrix(3, result, exp_labels)
        # plt.imshow(conf_matrix)
        # plt.colorbar()
        # plt.show()
        accs = []
        accs.append(BowDB.get_accuracy(0, result, exp_labels))
        accs.append(BowDB.get_accuracy(1, result, exp_labels))
        accs.append(BowDB.get_accuracy(2, result, exp_labels))

        avg_acc = BowDB.get_avg_accuracy_3class(result, exp_labels)
        print("Accuracy is: " + str(avg_acc))

        return [result, exp_labels, accs, avg_acc]
    else:
        db = BowDB(testImageDirName)
        print("Test set len of BowDB is " + str(len(db.get_all_data())))

        test_ftr_vectors, test_ranges = db.get_all_ftr_vecs(n_features=desc_n_features)

        test_histograms = BowDB.get_test_freq_hist(test_ftr_vectors, test_ranges, means)

        test_set = np.float32(test_histograms).reshape(-1, BowDB.n_bins)

        sc = StandardScaler()
        test_set_std = sc.fit_transform(test_set)

        result = svm.predict(test_set_std)[1]
            # plt.imshow(result)
            # plt.show()
        return [result, db.get_all_data()]

