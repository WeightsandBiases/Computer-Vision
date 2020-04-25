import os
import cv2
import h5py
import numpy as np
import tensorflow as tf
import logging as log


class HouseNumberDetector(object):
    def __init__(
        self, input_dir="", tf_model_dir="", model_filename="CNNHouseNumbersModel.h5"
    ):
        """class constructor of house number detector"""
        self.images = list()
        self.cnn = tf.keras.models.load_model(
            os.path.join(tf_model_dir, model_filename)
        )

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
        self, img, min_area=15, max_area=75, delta=20, min_diversity=10000,
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
            # add bounding boxes in a dictionary hashmap
            bounding_boxes.append(
                {"x_max": x_max, "x_min": x_min, "y_min": y_min, "y_max": y_max}
            )

        return bounding_boxes

    def get_img_pyramid(
        self,
        img,
        region,
        w_scales=[1.6, 1.4, 1.2, 1.0, 0.8, 0.6],
        h_scales=[1.4, 1.2, 1.0, 0.8, 0.6, 0.4],
        visualize=False,
    ):
        """
        generate bounding boxes of different scales for scale invariance
        Args:
            region(dict): dictionary of points of the original bounding box
            w_scales(list): list of floating point scales to scale the bounding
                            box width
            h_scales(list): list of floating point scales to scale the bounding
                            box height
        returns (list): list of dictionaries of the rescaled bounding boxes
        """
        scaled_regions = list()
        for i in range(len(h_scales)):
            scaled_region = dict()
            # width calculations
            height = region["y_max"] - region["y_min"]
            # CNN model is trained on a 32x32 square image
            # so width is set equal to height
            width = height
            width *= w_scales[i]
            scaled_region.update({"x_max": int(region["x_max"] + width / 2)})
            scaled_region.update({"x_min": int(region["x_min"] - width / 2)})
            # height calculations
            height *= h_scales[i]
            scaled_region.update({"y_max": int(region["y_max"] + height // 2)})
            scaled_region.update({"y_min": int(region["y_min"] - height // 2)})
            scaled_regions.append(scaled_region)
            # flag for visualizing the MSER regions
            if visualize:
                img = img.copy()
                COLOR = np.random.randint(0, high=255)
                cv2.rectangle(
                    img,
                    (scaled_region["x_min"], scaled_region["y_max"]),
                    (scaled_region["x_max"], scaled_region["y_min"]),
                    (COLOR, COLOR, COLOR),
                    1,
                )
                cv2.imshow("img", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return scaled_regions

    def get_best_pred(self, img, regions, visualize=False):
        """
        perform non maximal supression to choose the best region
        of prediction
        """
        img_pred = np.array([])
        for region in regions:
            img_pred = img[
                region["y_min"] : region["y_max"], region["x_min"] : region["x_max"]
            ]
            # reorder image to size 1, 32, 32 1
            img_pred = cv2.resize(img_pred, (32, 32))[:, :, np.newaxis, np.newaxis]
            # image is now 32, 32, 1, 1. Reorder to 1, 32, 32, 1.
            ORDER = (3, 0, 1, 2)
            img_pred = np.transpose(img_pred, ORDER)
            if visualize:
                cv2.imshow("img_pred", img_pred)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            print(self.cnn.predict(img_pred))
        print("--------------------------------------------")
    def detect_numbers(self):
        for img in self.images:
            regions = self.get_mser_regions(img)
            for region in regions:
                scaled_regions = self.get_img_pyramid(img, region)
                self.get_best_pred(img, scaled_regions)
