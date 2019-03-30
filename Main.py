# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
# from matplotlib import pyplot as plt

PATH = '/media/vale/Elements/Vale/Dataset Images'  # the path where the dataset is located


def compute_image_histogram(img_name, quadrant, use_mask=True):
    """
        Method that returns the histogram of the passed image. If 'use_mask' is set to True the method compute the
        histogram of the 'quadrant' only

        Parameters
        ----------
        img_name : str
            The path to the image that contains the identity document
        quadrant: list
            The list of coordinates [x,y] (top-left, top-right, bottom-right, bottom-left) of the identity document
            within the passed image
        use_mask: bool
            Specify if apply a mask to the image in order to consider the quadrant only. Default = True


        Returns
        -------
        histogram : cv2.hist
            The computed histogram
        """

    tmp_img = cv2.imread(img_name)
    img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2HSV)  # Convert the image from BGR to HSV format

    if use_mask:
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

    if use_mask:
        histogram = cv2.calcHist([img], [0, 1], mask, [180, 256], [0, 180, 0, 256])
    else:
        histogram = cv2.calcHist([img], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # plt.plot(histogram)
    # plt.show()

    return histogram


def get_position(dir_name, subdir_name, img_name):
    """
    Method that returns if the image fully contains a identity document and its position inside the image.
    (Image size 1080x1920)

    Parameters
    ----------
    dir_name : str
        The name of the 'global' directory that contains @img_name
    subdir_name : str
        The name of the directory inside @dir_name that contains @img_name
    img_name : str
        The name of the image

    Returns
    -------
    fully : bool
        True if the image fully contains a identity document
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
    Wrapper method that find for each folder inside the MIDV-500 dataset the best image (the one with the least
    Kullback-Leibler divergence wrt the base image of the document)

    Returns
    -------
    best_images : dict
        The dictionary that contains the selected images
    """

    document_directories = next(os.walk(PATH))[1]  # get the (50) 'global' directories
    best_images = dict.fromkeys(document_directories)  # dict. with the best image for each folder of each document

    for document in document_directories:
        path_img_dir = PATH + '/' + document + '/images'  # the 'images' directory of the current document
        img_directories = next(os.walk(path_img_dir))[1]  # get the (10) image directories of the document
        best_img_sub_dict = dict.fromkeys(img_directories)  # dict. with the best image for each folder
        base_histogram = compute_image_histogram(path_img_dir + '/' + document + '.tif', [],
                                                 False)  # compute the histogram of the base image, it will be used as
                                                         # reference for all the comparisisons

        for img_dir in img_directories:
            path_img = path_img_dir + '/' + img_dir
            images = os.listdir(path_img)  # get the (30) images inside the directory

            for img in images:
                fully_inside, quadrant = get_position(document, img_dir, img)
                if fully_inside:  # could be that a directory doesn't contain any image with a document fully within
                    hstgr = compute_image_histogram(path_img + '/' + img, quadrant)
                    comp_value = cv2.compareHist(base_histogram, hstgr, cv2.HISTCMP_KL_DIV)
                    if best_img_sub_dict[img_dir] is None:  # if there isn't already one image set as the best
                        best_img_sub_dict[img_dir] = [img, comp_value]
                    else:
                        best_sofar_value = best_img_sub_dict[img_dir][1]
                        if comp_value < best_sofar_value:
                            best_img_sub_dict[img_dir] = [img, comp_value]

        best_images[document] = best_img_sub_dict

    return best_images


dict_best = find_best_image()
print(json.dumps(dict_best, indent=2))
