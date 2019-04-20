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
DEFAULT_DICT_SIZE = 32

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
        length = round(len(self.__images)*0.9)
        return self.__images[:length]

    def get_test_set(self):
        length = round(len(self.__images)*0.9)
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
    def __make_hog_ftr_vector(image, n_feature):
        winSize = (64, 64)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        # compute(img[, winStride[, padding[, locations]]]) -> descriptors

        winStride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)
        hist = hog.compute(image, winStride=winStride, padding=padding, locations=locations)
        return hist, 1

    @staticmethod
    def __get_feature_vectors(samples, n_features):
        feature_vectors = []
        n_keypoints = 0
        desc_size = 0
        ranges = []
        for img in samples:
            ftr_vector, desc_size = BowDB.__make_hog_ftr_vector(img, n_features)
            ranges.append([n_keypoints, n_keypoints + ftr_vector.shape[0]])
            n_keypoints += ftr_vector.shape[0]
            feature_vectors.append(ftr_vector)

        descriptors = np.zeros([len(feature_vectors), ftr_vector.shape[0]], dtype=np.float32)
        i = 0

        for vec in feature_vectors:
            tvec = np.transpose(vec)
            descriptors[i, :] = tvec[:]
            i += 1
        return descriptors, ranges

    def get_train_ftr_vecs(self, n_features=DEFAULT_DESC_DIM):
        return BowDB.__get_feature_vectors(self.get_train_set(), n_features)

    def get_test_ftr_vecs(self, n_features=DEFAULT_DESC_DIM):
        return BowDB.__get_feature_vectors(self.get_test_set(), n_features)

    def get_all_ftr_vecs(self, n_features=DEFAULT_DESC_DIM):
        return BowDB.__get_feature_vectors(self.__images, n_features)

    @staticmethod
    def get_clasters(feature_vectors, is_test = False, means = 0):
        # feature_vectors = np.transpose(feature_vectors.astype(np.float32))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        if is_test:
            retval, labels, centers = cv2.kmeans(data=feature_vectors, K=BowDB.n_bins, bestLabels=None, centers=means, criteria=criteria, attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)

        else:
            retval, labels, centers = cv2.kmeans(data=feature_vectors, K=BowDB.n_bins, bestLabels=None, criteria=criteria, attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)

        return centers, labels

    @staticmethod
    def get_freq_hists(ftr_vectors, ranges, treshold=0, is_test=False, clasters_means=0):
        if is_test:
            clasters_means, labels = BowDB.get_clasters(ftr_vectors, is_test=True, means=clasters_means)
        else:
            clasters_means, labels = BowDB.get_clasters(ftr_vectors)

        samples_hist = np.zeros([len(ranges), BowDB.n_bins], dtype=np.int32)
        distances = np.zeros(ftr_vectors.shape[1], dtype=np.int32)
        max_dist = 0
        for i in range(len(ranges)):
            idx_from = ranges[i][0]
            idx_to = ranges[i][1]
            relevant_lbls = labels[idx_from:idx_to]
            if treshold > 0:
                curr_ftr_vectors = ftr_vectors[:, idx_from:idx_to]
                curr_claster = clasters_means[relevant_lbls[:]]
                dist = np.linalg.norm(np.transpose(np.transpose(curr_claster[:, 0]) - curr_ftr_vectors), axis=1)
                distances[idx_from:idx_to] = dist
                relevant_lbls = relevant_lbls[dist < treshold]
                # dist = dist[dist < treshold]
                # relevant_lbls = relevant_lbls[dist > 0]

            samples_hist[i, :] = np.histogram(relevant_lbls, bins=BowDB.n_bins, range=(0, BowDB.n_bins))[0]
        if treshold > 0:
            distances = distances[distances < treshold]
            # distances = distances[distances > 0]
            max_dist = max(distances[:])
            print("Max dist: " + str(max_dist))
        return samples_hist, clasters_means, distances, max_dist

    @staticmethod
    def get_test_freq_hist(ftr_vectors, range, treshold, clasters_means):
        return BowDB.get_freq_hists(ftr_vectors, range, treshold=treshold, is_test=True, clasters_means=clasters_means)

    @staticmethod
    def get_train_freq_hist(ftr_vectors, ranges, treshold=0):
        return BowDB.get_freq_hists(ftr_vectors, ranges=ranges, treshold=treshold)

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
    # Use a feature descriptor of your choice (SIFT, HOG, SURF etc) to extract feature vectors from all your DB Images.
    # Cluster the features into K clusters (use K-means, mean-shift or any clustering
    # method of your choice.K is a parameter ( or window size in mean shift etc).
    # This results in K visual words (usually numbered 1..K).
    # Decide how you will represent each word so that later ( in Step 3 and 5)
    # new features found in image will be tested against the visual words.
    # Another possible system parameter will be the complexity (dimension) of the feature descriptor.

    air_train_ftr_vectors, air_train_ranges = air_db.get_train_ftr_vecs(n_features=desc_n_features)
    moto_train_ftr_vectors, moto_train_ranges = moto_db.get_train_ftr_vecs(n_features=desc_n_features)
    elef_train_ftr_vectors, elef_train_ranges = elef_db.get_train_ftr_vecs(n_features=desc_n_features)

    BowDB.set_n_bins(dict_size)
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
    air_last_idx = air_train_ranges[len(air_train_ranges)-1][1]
    moto_last_idx = moto_train_ranges[len(moto_train_ranges)-1][1]
    for i in range(len(moto_train_ranges)):
        moto_train_ranges[i][0] += air_last_idx
        moto_train_ranges[i][1] += air_last_idx
    for i in range(len(elef_train_ranges)):
        elef_train_ranges[i][0] += moto_last_idx
        elef_train_ranges[i][1] += moto_last_idx

    train_ftr_vectors = np.concatenate((air_train_ftr_vectors, moto_train_ftr_vectors), axis=0)
    train_ftr_vectors = np.concatenate((train_ftr_vectors, elef_train_ftr_vectors), axis=0)

    train_ranges = air_train_ranges + moto_train_ranges + elef_train_ranges

    train_histograms, means, train_dists, air_train_max_dist = BowDB.get_train_freq_hist(train_ftr_vectors, train_ranges, treshold)
    # moto_train_histograms, moto_means, moto_train_dists, moto_train_max_dist = MotorbikeDB.get_train_freq_hist(moto_train_ftr_vectors, moto_train_ranges, treshold)
    # elef_train_histograms, elef_means, elef_train_dists, elef_train_max_dist = ElephantDB.get_train_freq_hist(elef_train_ftr_vectors, elef_train_ranges, treshold)
    #

    # air_train_X = 0.1 * air_db.label * np.ones((len(air_train_dists)))
    # moto_train_X = 0.1 * moto_db.label * np.ones((len(moto_train_dists)))
    # elef_train_X = 0.1 * elef_db.label * np.ones((len(elef_train_dists)))

    # plt.scatter(y=air_train_X , x=air_train_dists)
    # plt.scatter(y=moto_train_X , x=moto_train_dists)
    # plt.scatter(y=elef_train_X , x=elef_train_dists)
    # plt.legend()
    # plt.show()

    # 4)   Given the histograms (BOWs) of the DB image –
    #      build a classifier to distinguish between object and non-object.
    #      Build Classifier of your choice (SVM (linear or kernel), Fisher, Cascade etc).
    #
    # Make sure to save the classifier and / or its parameters to external file to be used later.

    # # in SVM::train @param trainData - training data that can be loaded from file using TrainData::loadFromCSV or .created with TrainData::create.
    # air_train_histogramssvm = cv2.ml_SVM()
    train_set = np.float32(train_histograms).reshape(-1, BowDB.n_bins)
    # moto_train_data = np.float32(moto_train_histograms).reshape(-1, BowDB.n_bins)
    # elef_train_data = np.float32(elef_train_histograms).reshape(-1, BowDB.n_bins)
    #
    # train_set = np.concatenate((air_train_data, moto_train_data), axis=0)
    # train_set = np.concatenate((train_set, elef_train_data), axis=0)

    air_responses = air_db.label * np.ones([len(air_db.get_train_set()), 1], dtype=np.int32)
    moto_responses = moto_db.label * np.ones([len(moto_db.get_train_set()), 1], dtype=np.int32)
    elef_responses = elef_db.label * np.ones([len(elef_db.get_train_set()), 1], dtype=np.int32)

    responses = np.concatenate((air_responses, moto_responses), axis=0)
    responses = np.concatenate((responses, elef_responses), axis=0)


    sc = StandardScaler()
    train_set_std = sc.fit_transform(train_set)

    # ts = dlib.vectors()
    # tl = dlib.array()
    # for i in range(train_set_std.shape[0]):
    #     ts.append(dlib.vector(train_set_std[i, :]))
    #     tl.append(responses[i])
    #     i += 1
    #
    # svm = dlib.svm_c_trainer_linear()
    # svm.be_verbose()
    # svm.train(ts, tl)


    # svm = SVC(C = 5.0, kernel ='rbf', degree = 3, gamma = 0.005, coef0 = 0.0, shrinking = True, probability = True, tol = 0.001, cache_size = 200, class_weight = None, verbose = False, max_iter = -1, decision_function_shape ='ovr', random_state = None)
    # svm.fit(X=train_set_std, y=responses)

    # # Saving the objects:
    # with open('svm.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump(svm, f)

    svm = cv2.ml_SVM.create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    # svm.setC(5.0)
    # svm.setGamma(1.0)

    svm.train(samples=train_set_std, layout=cv2.ml.ROW_SAMPLE, responses=responses)
    svm.save('svm_data.dat')

    # svm2 = SVC()
    # # Loading the objects:
    # if os.path.exists('svm.pkl'):
    #     with open('svm.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    #         svm2 = pickle.load(f)

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

    # svm = SVC()
    # # Loading the objects:
    # if os.path.exists('svm.pkl'):
    #     with open('svm.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    #         svm = pickle.load(f)

    means = np.float32([BowDB.n_bins])
    # Loading the objects:
    if os.path.exists('means.pkl'):
        with open('means.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
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
        # print(str(air_test_ftr_vectors))

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

        # print(test_ftr_vectors[:, 0])
        # print(test_ftr_vectors[:, (test_ftr_vectors.shape[1]-1)])
        # print(str(test_ranges))
        # print(len(test_ranges))

        test_histograms = BowDB.get_test_freq_hist(test_ftr_vectors, test_ranges, treshold, means)[0]
        # print(str(test_histograms[0]))
        # print(str(test_histograms[len(test_histograms)-1]))
        # air_test_X = 0.1 * air_db.label * np.ones((len(air_test_dists)))
        # moto_test_X = 0.1 * moto_db.label * np.ones((len(moto_test_dists)))
        # elef_test_X = 0.1 * elef_db.label * np.ones((len(elef_test_dists)))

        # plt.scatter(y=air_test_X , x=air_test_dists)
        # plt.scatter(y=moto_test_X , x=moto_test_dists)
        # plt.scatter(y=elef_test_X , x=elef_test_dists)
        # plt.legend()
        # plt.show()
        test_set = np.float32(test_histograms).reshape(-1, BowDB.n_bins)
        # air_test_data = np.float32(air_test_histograms).reshape(-1, BowDB.n_bins)
        # moto_test_data = np.float32(moto_test_histograms).reshape(-1, BowDB.n_bins)
        # elef_test_data = np.float32(elef_test_histograms).reshape(-1, BowDB.n_bins)

        # test_set = np.concatenate((air_test_data, moto_test_data), axis=0)
        # test_set = np.concatenate((test_set, elef_test_data), axis=0)

        air_exp_labels = air_db.label * np.ones([len(air_db.get_test_set()), 1], dtype=np.int32)
        moto_exp_labels = moto_db.label * np.ones([len(moto_db.get_test_set()), 1], dtype=np.int32)
        elef_exp_labels = elef_db.label * np.ones([len(elef_db.get_test_set()), 1], dtype=np.int32)

        exp_labels = np.concatenate((air_exp_labels, moto_exp_labels), axis=0)
        exp_labels = np.concatenate((exp_labels, elef_exp_labels), axis=0)

        sc = StandardScaler()

        test_set_std = sc.fit_transform(test_set)
        # result = test_svm.predict(test_set_std)[1]

        result = svm.predict(test_set_std)[1]
        # plt.scatter(np.array(range(len(result))), result)
        # plt.show()

        # score = svm.score(test_set_std, exp_labels)

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
        # print(str(test_ftr_vectors[:, 0]))
        # print(test_ftr_vectors[:, (test_ftr_vectors.shape[1]-1)])
        # print(str(test_ranges))
        # print(len(test_ranges))

        # test_histograms = np.zeros([len(test_ranges), db.n_bins], dtype=np.float32)

        test_histograms = BowDB.get_test_freq_hist(test_ftr_vectors, test_ranges, treshold, means)[0]

        # print(str(test_histograms[0]))
        # print(str(test_histograms[len(test_histograms) - 1]))
        test_set = np.float32(test_histograms).reshape(-1, BowDB.n_bins)

        sc = StandardScaler()
        test_set_std = sc.fit_transform(test_set)

        result = svm.predict(test_set_std)[1]
            # plt.imshow(result)
            # plt.show()
        return [result, db.get_all_data()]

