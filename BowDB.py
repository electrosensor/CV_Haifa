import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import pickle
import dlib

# 1)Create a DataBase –
# We will be using a set Database that can be found here.
# We will use the 3 classes of images: Airplane, Elephant and Motorbike
# They each have around 100 images each.The LAST 10 images are for testing.
#
# e) Put aside test image, (the last 10 images in the class(by name)).

DEFAULT_TRESHOLD = 840
DEFAULT_DESC_DIM = 116
DEFAULT_DICT_SIZE = 52

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
    def train_svm(X, y):

        svm = cv2.ml_SVM.create()
        svm.setKernel(cv2.ml.SVM_RBF)
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setC(60.0)
        svm.setGamma(0.0005)
        svm.train(samples=X, layout=cv2.ml.ROW_SAMPLE, responses=y)
        svm.save('svm_data.dat')

    @staticmethod
    def predict_svm(X):

        svm = cv2.ml_SVM.create()
        svm = svm.load('svm_data.dat')
        result = svm.predict(X)[1]
        return result

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
        act_labels = act_labels.astype(np.int32)
        exp_labels = exp_labels.astype(np.int32)
        hits = 0
        for i in range(len(act_labels)):
            if (act_labels[i] == checked_class) and (exp_labels[i] == checked_class):
                hits += 1
        accuracy = hits/len(act_labels)
        return accuracy

    @staticmethod
    def get_common_acc(act_labels, exp_labels):
        act_labels = act_labels.astype(np.int32)
        exp_labels = exp_labels.astype(np.int32)
        hits = 0
        for i in range(len(act_labels)):
            if act_labels[i] == exp_labels[i]:
                hits += 1
        accuracy = hits/len(act_labels)

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
        self.name = 'Airplane'


class MotorbikeDB(BowDB):
    def __init__(self):
        BowDB.__init__(self, "Datasets/Motorbike/")
        self.label = 1
        self.name = 'Motorbike'

class ElephantDB(BowDB):
    def __init__(self):
        BowDB.__init__(self, "Datasets/Elephant/")
        self.label = 2
        self.name = 'Elephant'

class ChairDB(BowDB):
    def __init__(self):
        BowDB.__init__(self, "Datasets/Chair/")
        self.label = 3
        self.name = 'Chair'

class FerryDB(BowDB):
    def __init__(self):
        BowDB.__init__(self, "Datasets/Ferry/")
        self.label = 4
        self.name = 'Ferry'

class WheelchairDB(BowDB):
    def __init__(self):
        BowDB.__init__(self, "Datasets/Wheelchair/")
        self.label = 5
        self.name = 'Wheelchair'

def train(class_list=[], treshold = DEFAULT_TRESHOLD, desc_n_features=DEFAULT_DESC_DIM, dict_size=DEFAULT_DICT_SIZE):

    BowDB.set_n_bins(dict_size)

    is_air = 'Airplane' in class_list
    is_moto = 'Motobike' in class_list
    is_elef = 'Elephant' in class_list
    is_chair = 'Chair' in class_list
    is_ferry = 'Ferry' in class_list
    is_wchair = 'Wheelchair' in class_list

    dbs = []
    if is_air:
        dbs.append(AirplaneDB())
    if is_moto:
        dbs.append(MotorbikeDB())
    if is_elef:
        dbs.append(ElephantDB())
    if is_chair:
        dbs.append(ChairDB())
    if is_ferry:
        dbs.append(FerryDB())
    if is_wchair:
        dbs.append(WheelchairDB())

    # 2)   Build visual word dictionary –

    all_train_feature_vecs = []
    all_train_ranges = []
    for db in dbs:
        print("Train set len of " + db.name + " is " + str(len(db.get_train_set())))
        train_ftr_vectors, train_ranges = db.get_train_ftr_vecs(n_features=desc_n_features)
        all_train_feature_vecs.append(train_ftr_vectors)
        all_train_ranges.append(train_ranges)

    # 3)   Create frequency histogram (BOW) for each of the images in the DB.

    for i in range(len(all_train_ranges)-1):
        last_idx = (all_train_ranges[i])[len(all_train_ranges[i]) - 1][1]
        for j in range(len(all_train_ranges[i+1])):
            (all_train_ranges[i+1])[j][0] += last_idx
            (all_train_ranges[i+1])[j][1] += last_idx

    # Assume 2 classes at least
    joined_ftr_vectors = np.concatenate((all_train_feature_vecs[0], all_train_feature_vecs[1]), axis=1)
    for fv in all_train_feature_vecs[2:]:
        joined_ftr_vectors = np.concatenate((joined_ftr_vectors, fv), axis=1)

    joined_train_ranges = []
    for tr in all_train_ranges:
        joined_train_ranges = joined_train_ranges + tr

    train_histograms, means = BowDB.get_train_freq_hist(joined_ftr_vectors, joined_train_ranges, treshold)

    # 4)   Given the histograms (BOWs) of the DB image

    train_set = np.float32(train_histograms).reshape(-1, BowDB.n_bins)

    responsesA = dbs[0].label * np.ones([len(dbs[0].get_train_set()), 1], dtype=np.int32)
    responsesB = dbs[1].label * np.ones([len(dbs[1].get_train_set()), 1], dtype=np.int32)
    joined_responces = np.concatenate((responsesA, responsesB), axis=0)
    for db in dbs[2:]:
        resp = db.label * np.ones([len(db.get_train_set()), 1], dtype=np.int32)
        joined_responces = np.concatenate((joined_responces, resp), axis=0)

    BowDB.train_svm(train_set, joined_responces)

    result = BowDB.predict_svm(train_set)

    # plt.scatter(np.array(range(len(result))), result)
    # plt.show()
    #
    # conf_matrix = BowDB.confusion_matrix(len(class_list), result, joined_responces)
    # plt.imshow(conf_matrix)
    # plt.colorbar()
    # plt.show()

 # Saving the objects:
    with open('means.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(means, f)

def test(class_list=[], treshold = DEFAULT_TRESHOLD, testImageDirName='', desc_n_features=DEFAULT_DESC_DIM, dict_size=DEFAULT_DICT_SIZE):

    BowDB.set_n_bins(dict_size)

    means = np.float32([BowDB.n_bins])
    # Loading the objects:
    if os.path.exists('means.pkl'):
        with open('means.pkl', 'rb') as f:
            means = pickle.load(f)

    if(testImageDirName is ''):

        is_air = 'Airplane' in class_list
        is_moto = 'Motobike' in class_list
        is_elef = 'Elephant' in class_list
        is_chair = 'Chair' in class_list
        is_ferry = 'Ferry' in class_list
        is_wchair = 'Wheelchair' in class_list

        dbs = []
        if is_air:
            dbs.append(AirplaneDB())
        if is_moto:
            dbs.append(MotorbikeDB())
        if is_elef:
            dbs.append(ElephantDB())
        if is_chair:
            dbs.append(ChairDB())
        if is_ferry:
            dbs.append(FerryDB())
        if is_wchair:
            dbs.append(WheelchairDB())

        # 2)   Build visual word dictionary –

        all_test_feature_vecs = []
        all_test_ranges = []
        for db in dbs:
            print("test set len of " + db.name + " is " + str(len(db.get_test_set())))
            test_ftr_vectors, test_ranges = db.get_test_ftr_vecs(n_features=desc_n_features)
            all_test_feature_vecs.append(test_ftr_vectors)
            all_test_ranges.append(test_ranges)

        # 3)   Create frequency histogram (BOW) for each of the images in the DB.

        for i in range(len(all_test_ranges) - 1):
            last_idx = (all_test_ranges[i])[len(all_test_ranges[i]) - 1][1]
            for j in range(len(all_test_ranges[i + 1])):
                (all_test_ranges[i + 1])[j][0] += last_idx
                (all_test_ranges[i + 1])[j][1] += last_idx

        # Assume 2 classes at least
        joined_ftr_vectors = np.concatenate((all_test_feature_vecs[0], all_test_feature_vecs[1]), axis=1)
        for fv in all_test_feature_vecs[2:]:
            joined_ftr_vectors = np.concatenate((joined_ftr_vectors, fv), axis=1)

        joined_test_ranges = []
        for tr in all_test_ranges:
            joined_test_ranges = joined_test_ranges + tr

        test_histograms = BowDB.get_test_freq_hist(joined_ftr_vectors, joined_test_ranges, means)

        test_set = np.float32(test_histograms).reshape(-1, BowDB.n_bins)

        yA = dbs[0].label * np.ones([len(dbs[0].get_test_set()), 1], dtype=np.int32)
        yB = dbs[1].label * np.ones([len(dbs[1].get_test_set()), 1], dtype=np.int32)
        expected_labels = np.concatenate((yA, yB), axis=0)
        for db in dbs[2:]:
            resp = db.label * np.ones([len(db.get_test_set()), 1], dtype=np.int32)
            expected_labels = np.concatenate((expected_labels, resp), axis=0)

        result = BowDB.predict_svm(test_set)
        # plt.scatter(np.array(range(len(result))), result)
        # plt.show()
        #
        # conf_matrix = BowDB.confusion_matrix(len(class_list), result, expected_labels)
        # plt.imshow(conf_matrix)
        # plt.colorbar()
        # plt.show()

        accs = []
        for i in range(6):
            accs.append(BowDB.get_accuracy(i, result, expected_labels))

        avg_acc = BowDB.get_common_acc(result, expected_labels)
        print("Accuracy is: " + str(avg_acc))

        return [result, expected_labels, accs, avg_acc]
    else:
        db = BowDB(testImageDirName)
        print("Test set len of BowDB is " + str(len(db.get_all_data())))

        test_ftr_vectors, test_ranges = db.get_all_ftr_vecs(n_features=desc_n_features)

        test_histograms = BowDB.get_test_freq_hist(test_ftr_vectors, test_ranges, means)

        test_set = np.float32(test_histograms).reshape(-1, BowDB.n_bins)

        result = BowDB.predict_svm(test_set)

        return [result, db.get_all_data()]

