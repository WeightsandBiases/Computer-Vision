import os
import cv2
import h5py
import numpy as np
import tensorflow as tf
import logging as log


class HouseNumberDetector(object):
    def __init__(
        self,
        input_dir="",
        output_dir="",
        tf_model_dir="",
        model_filename="CNNHouseNumbersModel.h5",
    ):
        """class constructor of house number detector"""
        self.images = list()
        self.images_color = list()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.init_class_logs()
        self.log.info("loading CNN House Numbers model")
        self.cnn = tf.keras.models.load_model(
            os.path.join(tf_model_dir, model_filename)
        )

    def init_class_logs(self):
        """
        initialize in class logging
        https://docs.python.org/3.1/library/logging.html
        """
        self.log = log.getLogger("CNNHouseNumbers")
        self.log.setLevel(log.INFO)
        ch = log.StreamHandler()
        ch.setLevel(log.INFO)
        formatter = log.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        self.log.addHandler(ch)

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
        H = 22
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

            img_color = cv2.imread(os.path.join(input_dir, img_file_path))

            # denoise image if flag is set to true
            if denoise:
                img = self.denoise_img(img)

            self.images.append(img)
            self.images_color.append(img_color)

    def get_mser_regions(
        self, img, min_area=15, max_area=250, delta=20, min_diversity=10000, visualize=True
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
            bb = {"x_max": x_max, "x_min": x_min, "y_min": y_min, "y_max": y_max}
            bounding_boxes.append(bb)
            # flag for visualizing the MSER regions
            if visualize:
                img = img.copy()
                COLOR = 0
                cv2.rectangle(
                    img,
                    (bb["x_min"], bb["y_max"]),
                    (bb["x_max"], bb["y_min"]),
                    (COLOR, COLOR, COLOR),
                    1,
                )
                cv2.imshow("img", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return bounding_boxes

    def get_overlapping_area(self, region_1, region_2):
        """
        calculates overlapping area of two regions
        Args: 
            region_1(dict): dictionary of x and y min max coordinates
            region_2(dict): dictionary of x and y min max coordinates
        Returns (int): overlapping area
        References:
        https://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
        """
        overlap_x_min = np.max((region_1["x_min"], region_2["x_min"]))
        overlap_x_max = np.min((region_1["x_max"], region_2["x_max"]))
        overlap_y_min = np.max((region_1["y_min"], region_2["y_min"]))
        overlap_y_max = np.min((region_1["y_max"], region_2["y_max"]))
        overlap_width = np.min((0, overlap_x_max - overlap_x_min))
        overlap_height = np.min((0, overlap_y_max - overlap_y_min))
        return overlap_width * overlap_height

    def get_average_regions(self, region_1, region_2):
        """
        averages the minimum and maximum values of two regions
        Args: 
            region_1(dict): dictionary of x and y min max coordinates
            region_2(dict): dictionary of x and y min max coordinates
        Returns (int): an averaged region
        """
        for key in region_1.keys():
            region_1[key] = int(np.average((region_1[key], region_2[key])))
        return region_1

    def non_max_supression(self, regions, area_threshold=2300):
        """
        Non Maximum Supression Malisiewicz et al.
        References:
        https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        Args:
            regions(list): list of dictionary of bounding box coordinates
            area_threshold: threshold for non max supression, larger values yields
                            more results
        """
        for i in range(len(regions) - 1):
            for j in range(1, len(regions)):
                # boundary checking because deleting regions list
                if i >= len(regions) or j >= len(regions):
                    break
                region_1 = regions[i]
                region_2 = regions[j]
                # if areas are overlapped too much
                if self.get_overlapping_area(region_1, region_2) > area_threshold:
                    # get the average of two regions and preserve that value
                    regions[i] = self.get_average_regions(region_1, region_2)
                    del regions[j]
        return regions

    def get_img_pyramid(
        self,
        img,
        region,
        w_scales=(1.6, 1.4, 1.2, 1.0, 0.8, 0.6),
        h_scales=(1.4, 1.2, 1.0, 0.8, 0.6, 0.4),
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

    def get_label_from_onehot(self, column_idx):
        """
        translate one hot encoded matrix to a 0-9 label
        Args: 
            column_idx(int): column index one hot encoded predictions
        Returns(int): image label 0 - 9
        """
        if column_idx >= 10:
            return 0
        else:
            return column_idx + 1

    def get_best_pred(self, img, regions, visualize=False, threshold=150):
        """
        perform non maximal supression to choose the best region
        of prediction
        Args:
            img (np.array): numpy image
            regions (list): list of dictionaries of bounding box coordinates
            threshold (float): threshold of a positive categorization
                               if threshold is not met, we do not predict a 
                               number
            visualize (bool): flag for displaying the image within bounding boxes

        """
        cnn_preds = list()
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
            cnn_preds.append(self.cnn.predict(img_pred).flatten())

        # convert predictions of all regions to a 2D array
        cnn_preds = np.asarray(cnn_preds)
        # find row and column of highest confidence
        col_max = np.argmax(np.max(cnn_preds, axis=0))
        row_max = np.argmax(np.max(cnn_preds, axis=1))
        # if bigger than predicted threshold
        if cnn_preds[row_max, col_max] > threshold:
            return (self.get_label_from_onehot(col_max), regions[row_max])
        else:
            return None

    def label_pred_image(self, image, pred, pred_region):
        """
        draws the bounding box and number predicted and saves the image
        """
        COLOR = (255, 255, 0)
        # Line thickness of 2 px
        SIZE = 1
        cv2.rectangle(
            image,
            (pred_region["x_min"], pred_region["y_max"]),
            (pred_region["x_max"], pred_region["y_min"]),
            COLOR,
            SIZE,
        )
        # font
        FONT = cv2.FONT_HERSHEY_PLAIN

        # org
        COORD = (pred_region["x_max"], pred_region["y_max"])

        # fontScale
        FONTSIZE = 1

        # Using cv2.putText() method
        image = cv2.putText(
            image, str(pred), COORD, FONT, FONTSIZE, COLOR, SIZE, cv2.LINE_AA
        )
        return image

    def save_pred_image(self, image, image_filename):
        cv2.imwrite(os.path.join(self.output_dir, image_filename), image)

    def detect_numbers(self):
        for idx, img in enumerate(self.images):
            regions = self.get_mser_regions(img)
            regions = self.non_max_supression(regions)
            for region in regions:
                scaled_regions = self.get_img_pyramid(img, region)
                pred_result = self.get_best_pred(img, scaled_regions)
                if pred_result:
                    pred, pred_region = pred_result
                    self.images_color[idx] = self.label_pred_image(
                        self.images_color[idx], pred, pred_region
                    )
            self.save_pred_image(self.images_color[idx], "predict_{}.png".format(idx))
