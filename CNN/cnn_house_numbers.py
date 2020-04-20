import os
import cv2
import h5py
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder


class CNNHouseNumbers(object):
    """
    A class to preprocess the images from the labeled dataset
    to classify numbers on buildings
    http://ufldl.stanford.edu/housenumbers/
    """

    def __init__(
        self,
        img_dir_path="",
        output_dir_path="",
        train_filename="train_32x32.mat",
        test_filename="test_32x32.mat",
        vald_filename="extra_32x32.mat",
    ):
        """
        Class constructor
        Args: 
            img_dir_path (str): directory path for test and train images
            train_filename (str): file name for the training images
            test_filename(str): file name for the testing images
        """
        self.img_dir_path = img_dir_path
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.vald_filename = vald_filename

    def load_mat_imgs(self, img_filename):
        """
        load matlab images
        Args:
            img_filename (str): file name of image
        Returns (tuple): a tuple of images (X) and their labels (y)
        """
        labeled_imgs = loadmat(os.path.join(self.img_dir_path, img_filename))
        return (labeled_imgs["X"], labeled_imgs["y"])

    def convert_imgs_to_gray(self, imgs):
        """
        batch convert images to grayscale
        Args:
            imgs(np.array): numpy array with dimensions 
                               images, rows, columns, channels
        returns (np.array): grayscale numpy array of images
        """
        gray_imgs = list()
        for img in imgs:
            gray_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        return np.asarray(gray_imgs)

    def mean_subtraction(self, imgs):
        """
        credit to https://cs231n.github.io/neural-networks-2/
        batch process mean subtraction on a group of images
        mean subtraction is simply taking the mean of the data then
        subtracting the mean from all the pixels of across the image
        set
        Args:
            imgs(np.array): numpy array of grayscale images
        Returns
            imgs(np.array): numpy array of grayscale images
        """
        # convert to float for mean subtraction
        imgs = imgs.astype(np.float64)
        # mean subtration
        imgs -= np.mean(imgs, axis=0)
        # convert back to uint8 image
        return imgs.astype(np.uint8)

    def data_normalization(self, imgs):
        """
        credit to https://cs231n.github.io/neural-networks-2/
        batch process data normalization to a group of images
        Args:
            imgs(np.array): numpy array of grayscale images
        Returns
            imgs(np.array): numpy array of grayscale images
        """
        # convert to float for data normalization
        imgs = imgs.astype(np.float64)
        # data normalization
        imgs /= np.std(imgs, axis=0)
        # convert back to uint8 image
        return imgs.astype(np.uint8)

    def one_hot_enc_labels(self, labels):
        """
        credit to 
        https://datascience.stackexchange.com/questions/30215/what-is-one-hot-encoding-in-tensorflow
        https://stackoverflow.com/questions/33681517/tensorflow-one-hot-encoder
        https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
        one hot encoding is used in machine learning to generate values
        for labels in a matrix in a way that will not affect the math that is
        performed.

        For example, our images are classified from 0-9.  
        When an algorithm interacts with these classifications a numerical
        9 may be confused to be greater than 1. 
        Thus we encode them in a matrix where the columns are from 0 to 9
        and the rows are of each image.  This way a value of 0 in the first
        image will be encoded in the schema below.
        ['label 1', 'label 2'... 'label 0]
        [ 0,         0,      ... 1       ]
        Args:
            labels (np.array): column array of integer labels (n x 1)
        Returns (np.array): a one hot encoded array of labels (n x n_labels)
        """

        encoded_labels = OneHotEncoder(sparse=False).fit_transform(labels)
        return encoded_labels

    def preprocess(self, save_preprocess=True, filename="CNN_PreProc_Dataset.h5"):
        """
        preprocess the data
        Args: 
            save_preprocess(bool): If true, proprocessed images
                                   and labels are saved in h5py format
            filename (str): If save_preprocess is true, then it will
                            be saved under the following filename
                            in h5py format
        """
        X_train, y_train = self.load_mat_imgs(self.train_filename)
        X_test, y_test = self.load_mat_imgs(self.test_filename)
        X_vald, y_vald = self.load_mat_imgs(self.vald_filename)
        # data is given in rows, columns, color channels, and
        # number of images
        # convert this to number of images, rows, columns, channels
        ORDER = (3, 0, 1, 2)  # re order of dimensions
        X_train = np.transpose(X_train, ORDER)
        X_test = np.transpose(X_test, ORDER)
        X_vald = np.transpose(X_vald, ORDER)
        # convert to grayscale
        X_train = self.convert_imgs_to_gray(X_train)
        X_test = self.convert_imgs_to_gray(X_test)
        X_vald = self.convert_imgs_to_gray(X_vald)
        # preprocess data using common data processing techniques
        X_train = self.mean_subtraction(X_train)
        X_test = self.mean_subtraction(X_test)
        X_vald = self.mean_subtraction(X_vald)
        X_train = self.data_normalization(X_train)
        X_test = self.data_normalization(X_test)
        X_vald = self.data_normalization(X_vald)
        # one hot encode labels
        y_train = self.one_hot_enc_labels(y_train)
        y_test = self.one_hot_enc_labels(y_test)
        y_vald = self.one_hot_enc_labels(y_vald)
        # option to save the preprocessed data
        if save_preprocess:
            # save preprocessed data in h5py format
            with h5py.File(filename, "w") as h5fd:
                h5fd.create_dataset("X_train", data=X_train)
                h5fd.create_dataset("y_train", data=y_train)
                h5fd.create_dataset("X_test", data=X_test)
                h5fd.create_dataset("y_test", data=y_test)
                h5fd.create_dataset("X_vald", data=X_vald)
                h5fd.create_dataset("y_vald", data=y_vald)
        else:
            # store data within class
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.X_vald = X_vald
            self.y_vald = y_vald

    def load_pre_processed_files(filepath):
        """
        load the pre processed data stored in h5py format
        Args:
            filepath(str): path of the file that is being used to store
            pre processed data
        """
        with h5py.File(filename, "r") as h5fd:
            self.X_train = h5fd["X_train"][:]
            self.y_train = h5fd["y_train"][:]
            self.X_test = h5fd["X_test"][:]
            self.y_test = h5fd["y_test"][:]
            self.X_vald = h5fd["X_vald"][:]
            self.y_vald = h5fd["y_vald"][:]

    def create_cnn(self, load):
        raise NotImplementedError
