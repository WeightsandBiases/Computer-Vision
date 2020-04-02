"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """
    x_flat_imgs = list()
    # labels are in the following format
    # subject##.xyz.png. We will use the number in ## as our label (“01” -> 1, “02” -> 2, etc.)
    y_labels = list()
    # subject is the first word in the file name
    SUBJECT_IDX = 0
    # label is the last two characters in the fie name
    LABEL_IDX = -2
    TOK = "."

    img_file_paths = [f for f in os.listdir(folder) if f.endswith(".png")]
    for img_file_path in img_file_paths:
        # read in image
        img = cv2.imread(os.path.join(folder, img_file_path))
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # resize image
        img = cv2.resize(img, tuple(size))
        # flatten image
        img = img.flatten()
        x_flat_imgs.append(img)
        # get image label
        label = img_file_path.split(TOK)[SUBJECT_IDX][LABEL_IDX:]
        y_labels.append(label)
    return (np.array(x_flat_imgs, dtype=np.uint8), np.array(y_labels, dtype=np.int8))


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    # length of X
    END = X.shape[0]
    # split point
    SPLIT = int(END * p)
    # randomize data
    random_indicies = np.random.permutation(END)
    X_rnd = X[random_indicies]
    y_rnd = y[random_indicies]
    # training data
    X_train = X_rnd[:SPLIT]
    y_train = y_rnd[:SPLIT]
    # test data
    X_test = X_rnd[SPLIT:]
    y_test = y_rnd[SPLIT:]
    return (X_train, y_train, X_test, y_test)


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    # axis is axis or axes along which the means are computed.
    # we want the mean of every column, thus axis is 0
    return np.mean(x, axis=0)


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """

    mu = get_mean_face(X)

    # compute sigma using equation from lectures
    sigma = (X - mu).T @ (X - mu)

    # compute eigens
    # eigen vector is the normalized (unit “length”) eigenvectors,
    # such that the column v[:,i] is the eigenvector corresponding to
    # the eigenvalue w[i].
    eigen_vals, eigen_vecs = np.linalg.eig(sigma)
    # sort by the eigen values in ascending order,
    # here argsort is used instead of sort
    # because we need to sort both eigen_value and eigen_vector by
    # eigen_value

    ascending_idx = eigen_vals.argsort()
    # we want the greatest eigenvalues, so we want to sort by descending
    # order, thus we need to reverse the array using python shorthand
    descending_idx = ascending_idx[::-1]
    # sort the eigen values and vectors in descending order (max first)
    eigen_vals = eigen_vals[descending_idx]
    eigen_vecs = eigen_vecs[:, descending_idx]
    # filter out just the top k eigen values and vectors
    eigen_vals = eigen_vals[:k]
    eigen_vecs = eigen_vecs[:, :k]

    return (eigen_vecs, eigen_vals)


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_data = len(self.Xtrain)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        # for each training stage...
        for i in range(self.num_iterations):
            # a) Renormalize the weights so they sum up to one
            self.weights /= np.sum(self.weights)

            # b) Instantiate the weak classifier h with the training data and labels.
            #    Train the classifier h
            #    Get predictions h(x) for all training examples
            wc = WeakClassifier(self.Xtrain, self.ytrain, self.weights)
            wc.train()
            h_X = list()
            for X in self.Xtrain:
                h_X.append(wc.predict(X))
            self.weakClassifiers.append(wc)

            # c) Find εj, summation of w_i where h(x_i) != y_i
            epsilon = 0
            for i in range(self.num_data):
                if self.ytrain[i] != h_X[i]:
                    epsilon += self.weights[i]

            # d) Calculate α_j
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            self.alphas.append(alpha)

            # e) If ε is greater than a (typically small) threshold:
            #    update the weights, otherwise stop the loop
            if epsilon > self.eps:
                for i in range(self.num_data):
                    if self.ytrain[i] != h_X[i]:
                        self.weights[i] *= np.exp(-self.ytrain[i] * alpha * h_X[i])
            else:
                break

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        correct = 0
        incorrect = 0
        predicts = self.predict(self.Xtrain)
        for i in range(self.num_data):
            if self.ytrain[i] == predicts[i]:
                correct += 1
            else:
                incorrect += 1
        return (correct, incorrect)

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        predicts = list()

        for i in range(len(X)):
            H_x = 0
            # predictions are a sum of all the weak classifiers
            for wc_i in range(len(self.weakClassifiers)):
                H_x += self.alphas[wc_i] * self.weakClassifiers[wc_i].predict(X[i])

            # assign predictions based on sign of H_x
            predicts.append(np.sign(H_x))
        return np.array(predicts)


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size
        # colors
        self.GRAY = 126
        self.WHITE = 255

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        height, width = self.size[:2]
        y_min, x_min = self.position[:2]
        # create a blank image
        img = np.zeros(shape, dtype=np.uint8)
        # dimension calculations
        half_height = int(height / 2)
        y_half = y_min + half_height
        y_max = y_min + height
        x_max = x_min + width

        # render the feature
        img[y_min:y_half, x_min:x_max] = self.WHITE
        img[y_half:y_max, x_min:x_max] = self.GRAY
        return img.astype(np.uint8)

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        height, width = self.size[:2]
        y_min, x_min = self.position[:2]
        # create a blank image
        img = np.zeros(shape, dtype=np.uint8)
        # dimension calculations
        half_width = int(width / 2)
        x_half = x_min + half_width
        x_max = x_min + width
        y_max = y_min + height

        # render the feature
        img[y_min:y_max, x_min:x_half] = self.WHITE
        img[y_min:y_max, x_half:x_max] = self.GRAY
        return img.astype(np.uint8)

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        height, width = self.size[:2]
        y_min, x_min = self.position[:2]
        # create a blank image
        img = np.zeros(shape, dtype=np.uint8)
        # dimension calculations
        one_third_height = int(height / 3)

        y_one_third = y_min + one_third_height
        y_two_third = y_min + one_third_height * 2

        x_max = x_min + width
        y_max = y_min + height

        # render the feature
        img[y_min:y_one_third, x_min:x_max] = self.WHITE
        img[y_one_third:y_two_third, x_min:x_max] = self.GRAY
        img[y_two_third:y_max, x_min:x_max] = self.WHITE
        return img.astype(np.uint8)

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        height, width = self.size[:2]
        y_min, x_min = self.position[:2]
        # create a blank image
        img = np.zeros(shape, dtype=np.uint8)
        # dimension calculations
        one_third_width = int(width / 3)

        x_one_third = x_min + one_third_width
        x_two_third = x_min + 2 * one_third_width
        x_max = x_min + width
        y_max = y_min + height

        # render the feature
        img[y_min:y_max, x_min:x_one_third] = self.WHITE
        img[y_min:y_max, x_one_third:x_two_third] = self.GRAY
        img[y_min:y_max, x_two_third:x_max] = self.WHITE
        return img.astype(np.uint8)

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        height, width = self.size[:2]
        y_min, x_min = self.position[:2]
        # create a blank image
        img = np.zeros(shape, dtype=np.uint8)
        # dimension calculations
        half_width = int(width / 2)
        half_height = int(height / 2)

        x_half = x_min + half_width
        y_half = y_min + half_height
        x_max = x_min + width
        y_max = y_min + height

        # render the feature
        img[y_min:y_half, x_min:x_half] = self.GRAY
        img[y_min:y_half, x_half:x_max] = self.WHITE
        img[y_half:y_max, x_min:x_half] = self.WHITE
        img[y_half:y_max, x_half:x_max] = self.GRAY
        return img.astype(np.uint8)

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def _get_score_two_horizontal_feature(self, ii):
        """
        Args:
            ii (numpy.array): Integral Image.

        Returns:
           float: Score value.
        """
        height, width = self.size[:2]
        y_min, x_min = self.position[:2]
        # dimension calculations, subtract one because
        # of integral image border
        y_min -= 1
        x_min -= 1
        half_height = int(height / 2)
        y_half = y_min + half_height
        y_max = y_min + height
        x_max = x_min + width

        # calculate white and gray areas
        A = ii[y_min][x_min]
        B = ii[y_min][x_max]
        C = ii[y_half][x_min]
        D = ii[y_half][x_max]
        white_area = A - B - C + D

        A = ii[y_half][x_min]
        B = ii[y_half][x_max]
        C = ii[y_max][x_min]
        D = ii[y_max][x_max]
        grey_area = A - B - C + D

        # add white area and subtract grey_area area
        score = white_area - grey_area
        return score

    def _get_score_two_vertical_feature(self, ii):
        """
        Args:
            ii (numpy.array): Integral Image.

        Returns:
           float: Score value.
        """
        height, width = self.size[:2]
        y_min, x_min = self.position[:2]
        # dimension calculations
        y_min -= 1
        x_min -= 1
        half_width = int(width / 2)
        x_half = x_min + half_width
        x_max = x_min + width
        y_max = y_min + height

        # calculate white and gray areas
        A = ii[y_min][x_min]
        B = ii[y_min][x_half]
        C = ii[y_max][x_min]
        D = ii[y_max][x_half]
        white_area = A - B - C + D

        A = ii[y_min][x_half]
        B = ii[y_min][x_max]
        C = ii[y_max][x_half]
        D = ii[y_max][x_max]
        grey_area = A - B - C + D

        # add white area and subtract grey_area area
        score = white_area - grey_area
        return score

    def _get_score_three_horizontal_feature(self, ii):
        """
        Args:
            ii (numpy.array): Integral Image.

        Returns:
           float: Score value.
        """
        height, width = self.size[:2]
        y_min, x_min = self.position[:2]
        # dimension calculations
        y_min -= 1
        x_min -= 1
        one_third_height = int(height / 3)

        y_one_third = y_min + one_third_height
        y_two_third = y_min + one_third_height * 2

        x_max = x_min + width
        y_max = y_min + height

        # calculate white and gray areas
        A = ii[y_min][x_min]
        B = ii[y_min][x_max]
        C = ii[y_one_third][x_min]
        D = ii[y_one_third][x_max]
        white_area = A - B - C + D

        A = ii[y_one_third][x_min]
        B = ii[y_one_third][x_max]
        C = ii[y_two_third][x_min]
        D = ii[y_two_third][x_max]

        grey_area = A - B - C + D

        A = ii[y_two_third][x_min]
        B = ii[y_two_third][x_max]
        C = ii[y_max][x_min]
        D = ii[y_max][x_max]

        white_area += A - B - C + D

        # add white area and subtract grey_area area
        score = white_area - grey_area
        return score

    def _get_score_three_vertical_feature(self, ii):
        """
        Args:
            ii (numpy.array): Integral Image.

        Returns:
           float: Score value.
        """
        height, width = self.size[:2]
        y_min, x_min = self.position[:2]
        # dimension calculations
        y_min -= 1
        x_min -= 1
        one_third_width = int(width / 3)

        x_one_third = x_min + one_third_width
        x_two_third = x_min + 2 * one_third_width
        x_max = x_min + width
        y_max = y_min + height

        # calculate white and gray areas
        A = ii[y_min][x_min]
        B = ii[y_min][x_one_third]
        C = ii[y_max][x_min]
        D = ii[y_max][x_one_third]

        white_area = A - B - C + D

        A = ii[y_min][x_one_third]
        B = ii[y_min][x_two_third]
        C = ii[y_max][x_one_third]
        D = ii[y_max][x_two_third]

        grey_area = A - B - C + D

        A = ii[y_min][x_two_third]
        B = ii[y_min][x_max]
        C = ii[y_max][x_two_third]
        D = ii[y_max][x_max]

        white_area += A - B - C + D

        # add white area and subtract grey_area area
        score = white_area - grey_area
        return score

    def _get_score_four_square_feature(self, ii):
        """
        Args:
            ii (numpy.array): Integral Image.

        Returns:
           float: Score value.
        """
        height, width = self.size[:2]
        y_min, x_min = self.position[:2]
        # dimension calculations
        y_min -= 1
        x_min -= 1
        half_width = int(width / 2)
        half_height = int(height / 2)

        x_half = x_min + half_width
        y_half = y_min + half_height
        x_max = x_min + width
        y_max = y_min + height

        # calculate white and gray areas
        A = ii[y_min][x_min]
        B = ii[y_min][x_half]
        C = ii[y_half][x_min]
        D = ii[y_half][x_half]

        grey_area = A - B - C + D

        # TODO finish!


    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        ii = ii.astype(np.float32)

        # two horizontal feature
        if self.feat_type == (2, 1):
            return self._get_score_two_horizontal_feature(ii)
        # two vertical feature
        elif self.feat_type == (1, 2):
            return self._get_score_two_vertical_feature(ii)
        # three horizontal feature
        elif self.feat_type == (3, 1):
            return self._get_score_three_horizontal_feature(ii)
        # three vertical feature
        elif self.feat_type == (1, 3):
            return self._get_score_two_horizontal_feature(ii)
        else:
            raise NotImplementedError


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    integral_imgs = list()

    for img in images:
        # sum over the rows and columns
        integral_img = np.cumsum(np.cumsum(img, axis=0), axis=1)
        integral_imgs.append(integral_img)

    return integral_imgs


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """

    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1 * np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {
            "two_horizontal": (2, 1),
            "two_vertical": (1, 2),
            "three_horizontal": (3, 1),
            "three_vertical": (1, 3),
            "four_square": (2, 2),
        }

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(
                                    feat_type, [posi, posj], [sizei - 1, sizej - 1]
                                )
                            )
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = (
            np.ones(len(self.posImages), dtype="float")
            * 1.0
            / (2 * len(self.posImages))
        )
        weights_neg = (
            np.ones(len(self.negImages), dtype="float")
            * 1.0
            / (2 * len(self.negImages))
        )
        weights = np.hstack((weights_pos, weights_neg))

        print(" -- select classifiers --")
        for i in range(num_classifiers):

            # TODO: Complete the Viola Jones algorithm

            raise NotImplementedError

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).

        for x in scores:
            # TODO
            raise NotImplementedError

        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """

        raise NotImplementedError
