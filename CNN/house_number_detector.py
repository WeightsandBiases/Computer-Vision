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

    def read_imgs(self, input_dir='input_images'):
        """
        read input images in as grayscale images
        """
        img_file_paths = [f for f in os.listdir(input_dir) if f.endswith(".png")]
        for img_file_path in img_file_paths:
            # read in image
            self.images.append(
                cv2.imread(
                    os.path.join(input_dir, img_file_path), cv2.IMREAD_GRAYSCALE
                )
            )


    def get_mser_regions(self, img, min_area=50, max_area=250, delta=12):
        """
        find regions in images where the is likely to be a number
        References:
        https://stackoverflow.com/questions/17647500/exact-meaning-of-the-parameters-given-to-initialize-mser-in-opencv-2-4-x
        https://kite.com/python/docs/cv2.MSER
        https://github.com/Belval/opencv-mser/blob/master/mser.py
        """
        img = img.copy()
        mser = cv2.MSER_create(_min_area=min_area, _max_area=max_area, _delta=delta)
        regions, _ = mser.detectRegions(img)
        for p in regions:
            xmax, ymax = np.amax(p, axis=0)
            xmin, ymin = np.amin(p, axis=0)
            cv2.rectangle(img, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
            
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def detect_numbers(self):
        for img in self.images:
            self.get_mser_regions(img)