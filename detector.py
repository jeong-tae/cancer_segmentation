import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import _pickle as cPickle
import random

class Detector(object):
    def __init__(self, filelist = "/home/jtlee/workspace/canser_segmentation/data/frames.txt", ckpt = None):

        self.n_jobs = 1
        self.n_estimators = 10 # hyper-param
        self.context_size = 3 # hyper-param
        self.rf = RandomFroestClassifier(n_jobs = self.n_jobs, n_estimators = self.n_estimators)

        if ckpt != None:
            raise NotImplementedError("Not implemented behavior!")

        self.dir_path = '/home/jtlee/workspace/canser_segmentation/data/'
        file_paths = np.array(open(filelist).readlines())

        random.seed(1004)
        np.random.seed(1004)
        random.shuffle(file_path)

        split = np.random.uniform(0, 1, len(file_paths)) <= .9

        self.train_files = file_paths[split == True]
        self.test_files = file_paths[split == False]

        print(" [*] Number of observations in the training data:", len(self.train_files))
        print(" [*] Number of observations in the test data:", len(self.test_files))

    def context_feature(self, image, size = 3):
        if size % 2 == 0:
            raise NotImplementedError(" [!] Currently even size of context feature is not supporting!")
        padding = int(size / 2)
        shape = image.shape
        pad_image = np.zeros((shape[0] + padding*2, shape[1] + padding*2))
        pad_image[padding:-padding, padding:-padding] = image

        features = []
        for i in range(padding, shape[0]+padding):
            for j in range(padding, shape[1]+padding):
                feature = pad_image[i-padding:i+padding, j-padding:j+padding].reshape((-1))
                features.append(feature)

        features = np.concatenate(features, axis = 0)
        features = features.reshape((-1, size**2))
        return features

    def train(self):
        images = []
        gt_images = []
        for i in tqdm(range(len(self.train_files))):
            paired = self.train_files[i].split('\t')
            img = cv2.imread(self.dir_path + paired[0].strip())
            flat = self.context_feature(img, size = self.context_size)
            gt_img = cv2.imread(self.dir_path + paired[1].strip())

            gt_img[gt_img > 0] = 1 # binarization
            gt_flat = gt_img.reshape((-1))

            images.append(flat)
            gt_images.append(gt_flat)
        images = np.concatenate(images, axis = 0)
        gt_images = np.concatenate(gt_images, axis = 0)
        print(" [*] Start to learn... it may take time long")
        self.rf.fit(images, gt_images)
        print(" [*] Train done")

    def validation(self):
        print(" [*] Validation...")
        images = []
        gt_images = []
        for i in tqdm(range(len(self.test_files))):
            paired = self.test_files[i].split('\t')
            img = cv2.imread(self.dir_path + paired[0].strip())
            flat = self.context_featue(img, size = self.context_size)
            gt_img = cv2.imread(self.dir_path + paired[1].strip())

            gt_img[gt_img > 0] = 1
            gt_flat = gt_img.reshape((-1))

            images.append(flat)
            gt_images.append(gt_flat)

        images = np.concatenate(images, axis = 0)
        gt_images = np.concatenate(gt_images, axis = 0)

        pred = self.rf.predict(images)
        prec, recall, f1, _ = precision_recall_fscore_support(gt_images, images, average = 'binary')
        acc = (sum(gt_images == pred) / gt_flat.shape[0])

        print(" [*] acc: %.3f, prec: %.2f, recall: %.2f, f1: %.3f"%(acc, prec, recall, f1))
        return f1

    def segment_mask(self, shape, image, label):
        # shape, shape: [width, hegiht] of input image
        # label, shape: [n,]
        mask = np.full([label.shape[0], 3], 0, dtype = 'uint8')
        indices = np.where(label == 1)

        mask[indices] = image[indices]
        mask = mask.reshape(list(shape))
        return np.array(mask)

    def inference(self, image_file):
        img = cv2.imread(image_file) # gray
        img_shape = img.shape
        flat = self.context_featue(img, size = self.context_size)

        pred = self.rf.predict(flat)
        
        mask = self.segment_mask(img_shape, img, pred)
        plt.figure()
        plt.axis('off')
        plt.imshow(mask)
        plt.show()

    def save(self, model, path = "/home/jtlee/workspace/cancer_segmentation/ckpt/", n = 0):
        path = path + "ckpt_" + str(n) + ".pkl"
        f = open(path, 'wb')
        cPickle.dump(model, f)
        f.close()
        print(" [*] model is saved at %s"%path)
