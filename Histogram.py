# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt


PATH = '/media/vale/HDDVale/Uni/DataSet'  # the path where the dataset is located


def compute_image_histogram(img_name, quadrangle, use_mask=True):
    """
    Method that returns the histogram of the passed image. If 'use_mask' is set to True the method compute the
    histogram of the 'quadrangle' only

    Parameters
    ----------
    img_name : str
        The path to the image that contains the identity document
    quadrangle: list
        The list of coordinates [x,y] (top-left, top-right, bottom-right, bottom-left) of the identity document
        within the passed image
    use_mask: bool
        Specify if apply a mask to the image in order to consider the quadrangle only. Default = True

    Returns
    -------
    histogram : cv2.hist
        The computed histogram
    """

    tmp_img = cv2.imread(img_name)
    img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2HSV)  # Convert the image from BGR to HSV format

    if use_mask:
        # get the coordinate of the 4 vertices of the identity document (tl -> top left, br -> bottom right, ...)
        x_tl = quadrangle[0][0]
        y_tl = quadrangle[0][1]
        x_tr = quadrangle[1][0]
        y_tr = quadrangle[1][1]
        x_br = quadrangle[2][0]
        y_br = quadrangle[2][1]
        x_bl = quadrangle[3][0]
        y_bl = quadrangle[3][1]

        # create and apply the mask (and visualize)
        rect = [[x_br, y_br], [x_bl, y_bl], [x_tl, y_tl], [x_tr, y_tr]]
        poly = np.array([rect], dtype=np.int32)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, poly, 255)
        # masked_img = cv2.bitwise_and(img, img, mask=mask)
        # plt.imshow(masked_img)
        # plt.title(img_name[-12:])
        # plt.show()

    if use_mask:
        histogram = cv2.calcHist([img], [0, 1], mask, [180, 256], [0, 180, 0, 256])
    else:
        histogram = cv2.calcHist([img], [0, 1], None, [180, 256], [0, 180, 0, 256])

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
    quadrangle : list
        The list of coordinates [x,y] (top-left, top-right, bottom-right, bottom-left) of the identity document
    """

    path_json = PATH + '/' + dir_name + '/ground_truth' + '/' + subdir_name + '/'
    file_name = path_json + img_name[:-4] + '.json'

    fully_inside = False
    quadrangle = []  # the quadrant that contains the identity document
    i = 0  # counter of vertices satisfying the constraints
    with open(file_name) as f:
        data = json.load(f)
        quadrangle = data['quad'].copy()
        for xy in quadrangle:
            if (xy[0] >= 0) and (xy[0] <= 1080):
                if xy[1] >= 0 and (xy[1] <= 1920):
                    i += 1

    if i == 4:  # all 4 vertices are within the image
        fully_inside = True

    return fully_inside, quadrangle


def compute_distances():
    """
    Method that for each images (that fully contains a document) inside the MIDV-500 dataset compute the
    Kullback-Leibler divergence wrt the base image of the document.

    Returns
    -------
    kl_distances : dict
        The dictionary that contains all the KL divergences computed
    """

    document_directories = next(os.walk(PATH))[1]  # get the (50) 'global' directories
    kl_distances = dict.fromkeys(document_directories)  # global dictionary

    for document in document_directories:
        path_img_dir = PATH + '/' + document + '/images'  # the 'images' directory of the current document
        img_directories = next(os.walk(path_img_dir))[1]  # get the (10) image directories of the document
        distances_img_sub_dict = dict.fromkeys(img_directories)  # sub-dictionary
        base_histogram = compute_image_histogram(path_img_dir + '/' + document + '.tif', [],
                                                 False)  # compute the histogram of the base image
        for img_dir in img_directories:
            path_img = path_img_dir + '/' + img_dir
            images = os.listdir(path_img)  # get the (30) images inside the directory
            distance_values = []  # list of kl distances (tuples of(image name, KL distance))
            for img in images:
                fully_inside, quadrangle = get_position(document, img_dir, img)
                if fully_inside:  # check if the image fully contains a document
                    hstgr = compute_image_histogram(path_img + '/' + img, quadrangle)  # compute histogram
                    comp_value = cv2.compareHist(base_histogram, hstgr, cv2.HISTCMP_KL_DIV)  # compute difference
                    distance_values.append((img, comp_value))  # append to the list
            distances_img_sub_dict[img_dir] = distance_values

        kl_distances[document] = distances_img_sub_dict

    return kl_distances


def find_best_images(kl_dist):
    """
    Method that return the images that have a Kullback-Leibler divergence lower than a fixed threshold
    (the first quantile)

    Parameters
    ----------
    kl_dist : dict
        The dictionary containing the KL divergences stored as tuples of(image name, KL distance)

    Returns
    -------
    best_images : dict
        The dictionary that contains the selected images
    """
    best_images = dict.fromkeys(kl_dist.keys())  # new global dictionary (same structure as the passed one)

    for k1, v1 in kl_dist.items():  # iterate through the passed dictionary
        best_img_sub_dict = {}  # new sub-dictionary (same structure as the passed one)
        for k2, v2 in v1.items():   # iterate through the sub-dictionary (of the passed dict.)
            best_img_sub_dict[k2] = []

            if len(v2) > 0:     # if the list is not empty
                distances = []  # list of all distances
                for i in v2:
                    distances.append(i[1])      # get (and append) the value of kl distance
                threshold = np.quantile(distances, 0.25)    # compute the first quantile of the list of distances
                for j in v2:
                    if j[1] <= threshold:       # if the distance is lower than the threshold
                        best_img_sub_dict[k2].append(j)     # add to the new sub-dictionary the current tuple

        best_images[k1] = best_img_sub_dict

    return best_images


kl_dict = compute_distances()
best_dict = find_best_images(kl_dict)

print(json.dumps(best_dict, indent=2))
