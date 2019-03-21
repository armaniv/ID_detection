#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
#from matplotlib import pyplot as plt

PATH = '/home/vale/Documenti/Project Course/DataSet'  # the path where the dataset is located


def compute_image_histogram(img_name, quadrant):
    """
        Method that returns the histogram of the identity document within the passed image.

        Parameters
        ----------
        img_name : str
            The path to the image that contains the identity document
        quadrant: list
            The list of coordinates [x,y] (top-left, top-right, bottom-right, bottom-left) of the identity document
            within the passed image

        Returns
        -------
        histogram : cv2.hist
            The histogram of the identity document within the passed image
        """

    img = cv2.imread(img_name, -1)  # imread unchanged

    # get the coordinate of the 4 vertices of the identity document (tl -> top left, br -> bottom right, ...)
    x_tl = quadrant[0][0]
    y_tl = quadrant[0][1]
    x_tr = quadrant[1][0]
    y_tr = quadrant[1][1]
    x_br = quadrant[2][0]
    y_br = quadrant[2][1]
    x_bl = quadrant[3][0]
    y_bl = quadrant[3][1]

    # get the max height and width (necessary for any rotated documents)
    w = max(abs(x_tl - x_br), abs(x_bl - x_tr))
    h = max(abs(y_tl - y_br), abs(y_bl - y_tr))

    # get the coordinates where start to cut off (necessary for any rotated documents)
    x_start = min(x_tl, x_bl)
    y_start = min(y_tl, y_tr)

    # create a mask with the computed values
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[y_start:y_start + h, x_start:x_start + w] = 255

    color = ('b', 'g', 'r')
    for channel, col in enumerate(color):
        histogram = cv2.calcHist([img], [channel], mask, [256], [0, 256])
        # plt.plot(histogram, color=col)
        # plt.xlim([0, 256])

    # plt.show()

    return histogram


def get_position(dir_name, subdir_name, img_name):
    """
    Method that returns if the image fully contains the identity document and its position inside it.
    (Image size 1080x1920)

    Parameters
    ----------
    dir_name : str
        The name of the 'global' directory that contains @img_name
    subdir_name : str
        The name of the directory inside @dir_name that contains @img_name
    img_name : str
        The name of the image that is checked if fully contains the identity document

    Returns
    -------
    fully : bool
        True if the image fully contains the identity document
    quadrant : list
        The list of coordinates [x,y] (top-left, top-right, bottom-right, bottom-left) of the identity document
    """

    path_json = PATH + '/' + dir_name + '/ground_truth' + '/' + subdir_name + '/'
    file_name = path_json + img_name[:-4] + '.json'

    fully_inside = False
    quadrant = []  # the quadrant that contains the identity document
    i = 0  # counter of vertices satisfying the constraints
    with open(file_name) as f:
        data = json.load(f)
        quadrant = data['quad'].copy()
        for xy in quadrant:
            if (xy[0] >= 0) and (xy[0] <= 1080):
                if xy[1] >= 0 and (xy[1] <= 1920):
                    i += 1

    if i == 4:  # all 4 vertices are within the image
        fully_inside = True

    return fully_inside, quadrant


def find_best_image():
    """
    Wrapper method that find for each folder inside the MIDV-500 dataset the best image (the one with the most balanced
    histogram)
    """

    document_directories = next(os.walk(PATH))[1]  # get the (50) 'global' directories
    best_images = dict.fromkeys(document_directories)  # dict. with the best image for each folder of each document

    for document in document_directories:
        path_img_dir = PATH + '/' + document + '/images'  # the 'images' directory of the current document
        img_directories = next(os.walk(path_img_dir))[1]  # get the (10) image directories of the document
        best_img_sub_dict = dict.fromkeys(img_directories)  # dict. with the best image for each folder

        for img_dir in img_directories:
            path_img = path_img_dir + '/' + img_dir
            images = os.listdir(path_img)  # get the (30) images inside the directory

            for img in images:
                fully_inside, quadrant = get_position(document, img_dir, img)
                if fully_inside:    # could be that a directory doesn't contain any image with a document fully within
                    if best_img_sub_dict[img_dir] is None:  # if there isn't already one image set as the best
                        hstgr = compute_image_histogram(path_img + '/' + img, quadrant)
                        best_img_sub_dict[img_dir] = [img, hstgr]
                    else:
                        hstgr = compute_image_histogram(path_img + '/' + img, quadrant)
                        best_sofar_hstgr = best_img_sub_dict[img_dir][1]
                        # TODO choose best histogram

        best_images[document] = best_img_sub_dict

    # print in a readable way the dictionary that contains the best images
    for key in best_images.keys():
        print(key, end=' -> {')
        for key2, value in best_images[key].items():
            if value is None:   # if no best image for the current folder is present
                print(key2 + ':', '--', end=', ')
            else:
                print(key2 + ':', value[0], end=', ')
        print('}')


find_best_image()
