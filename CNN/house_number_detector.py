import os
import cv2
import h5py
import numpy as np
import tensorflow as tf
import logging as log


class HouseNumberDetector(object):
    def __init__(self, input_dir=""):
        """class constructor of house number detector"""
        self.images = list()

    def denoise_img(self, img):
        """
        handle image noise
        Args: 
            img(np.array): numpy array of image
        Returns(np.array): numpy array 
        """
        # denoising parameters
        # from the openCv docs
        # Parameter regulating filter strength for luminance component.
        # Bigger h value perfectly removes noise but also removes image details,
        # smaller h value preserves details but also preserves some noise
        H = 25
        # Size in pixels of the template patch that is used to compute weights.
        # Should be odd. Recommended value 7 pixels
        TEMPLATE_WINDOW_SIZE = 7
        # Size in pixels of the window that is used to compute weighted average for
        # given pixel. Should be odd. Affect performance linearly:
        # greater searchWindowsSize - greater denoising time.
        # Recommended value 21 pixels
        SEARCH_WINDOW_SIZE = 21

        return cv2.fastNlMeansDenoising(
            img, None, H, TEMPLATE_WINDOW_SIZE, SEARCH_WINDOW_SIZE
        )

    def read_imgs(self, input_dir="input_images", denoise=True):
        """
        read input images in as grayscale images
        """
        img_file_paths = [f for f in os.listdir(input_dir) if f.endswith(".png")]
        for img_file_path in img_file_paths:
            # read in image

            img = cv2.imread(
                os.path.join(input_dir, img_file_path), cv2.IMREAD_GRAYSCALE
            )

            # denoise image if flag is set to true
            if denoise:
                img = self.denoise_img(img)

            self.images.append(img)

    def get_mser_regions(
        self,
        img,
        min_area=15,
        max_area=75,
        delta=20,
        min_diversity=5000,
        padding_h=6,
        padding_w=4,
        visualize=True,
    ):
        """
        find regions in images where the is likely to be a number
        Args:
            img(np.array): image to process
            min_area(int): the minimum area of a region containing a number
            max_area(int): the maximum area of a region containing a number
            delta(float): the delta threshold used in MSER region finding
            min_diversity(float): threshold to prune regions that are too similar
            padding_h(int): number of pixels to pad the height of the bounding box
            padding_w(int): number of pixels to pad the width of the bounding box
        References:
        https://stackoverflow.com/questions/17647500/exact-meaning-of-the-parameters-given-to-initialize-mser-in-opencv-2-4-x
        https://kite.com/python/docs/cv2.MSER
        https://github.com/Belval/opencv-mser/blob/master/mser.py
        """
        bounding_boxes = list()
        img = img.copy()
        mser = cv2.MSER_create(
            _min_area=min_area,
            _max_area=max_area,
            _delta=delta,
            _min_diversity=min_diversity,
        )
        regions, _ = mser.detectRegions(img)
        for p in regions:
            x_max, y_max = np.amax(p, axis=0)
            x_min, y_min = np.amin(p, axis=0)
            # apply padding to bounding box
            x_max += padding_w // 2
            x_min -= padding_w // 2
            y_max += padding_h // 2
            y_min -= padding_h // 2
            # add bounding boxes in a dictionary hashmap
            bounding_boxes.append(
                {"x_max": x_max, "x_min": x_min, "y_min": y_min, "y_max": y_max}
            )
            # flag for visualizing the MSER regions
            if visualize:
                COLOR = np.random.randint(0, high=255)
                cv2.rectangle(
                    img, (x_min, y_max), (x_max, y_min), (COLOR, COLOR, COLOR), 1
                )

        # flag for visualizing the MSER regions
        if visualize:
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return bounding_boxes

    def detect_numbers(self):
        for img in self.images:
            regions = self.get_mser_regions(img)
            