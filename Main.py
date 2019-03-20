#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# istogramma -> La parte sinistra dell'asse orizzontale rappresenta le aree scure e nere, la parte in mezzo le aree
# grigie e la parte destra le aree bianche e più chiare. L'asse verticale rappresenta la grandezza dell'area che è stata
# catturata in ognuna di queste zone. Così l'istogramma di un'immagine molto luminosa con poche aree scure e/o ombre
# avrà più punti verso la destra e il centro del grafico, al contrario per un'immagine poco luminosa.

import os
import json

PATH = '/home/vale/Documenti/Project Course/DataSet'  # the path where the dataset is located


def check_fully_inside(dir_name, subdir_name, img_name):
    """
    Method that search the json of the passed image and checks if the images fully contains the identity document.
    The image size is w=1080 and h=1920. Coordinates are given as a pairt [x, y] denoted in pixels int the order:
    top-left, top-right, bottom-right, bottom-left

    Parameters
    ----------
    dir_name : str
        the name of the 'global' directory that contains @img_name
    subdir_name : str
        the directory inside @dir_name that contains @img_name
    img_name : str
        the image that is checked if fully contains the identity document

    Returns
    -------
    boolean
        True if the image fully contains the identity document
    """

    path_json = PATH + '/' + dir_name + '/ground_truth' + '/' + subdir_name + '/'
    file_name = path_json + img_name[:-4] + '.json'

    fully = False  # fully inside
    i = 0  # counter of vertices satisfying the constraints
    with open(file_name) as f:
        data = json.load(f)
        for xy in data['quad']:
            if (xy[0] >= 0) and (xy[0] <= 1080):
                if xy[1] >= 0 and (xy[1] <= 1920):
                    i += 1

    if i == 4:  # all 4 vertices are within the image
        fully = True

    return fully


def find_best_image():
    """
    Wrapper method that find for each folder inside the MIDV-500 dataset the best image (the one with the most balanced
    histogram)
    """

    document_directories = next(os.walk(PATH))[1]  # get the (50) 'global' directories

    for document in document_directories:
        path_img_dir = PATH + '/' + document + '/images'  # the 'images' directory of the current document
        img_directories = next(os.walk(path_img_dir))[1]  # get the (10) image directories of the document
        for img_dir in img_directories:
            images = os.listdir(path_img_dir + '/' + img_dir)  # get the (30) images inside the directory
            for img in images:
                if check_fully_inside(document, img_dir, img):
                    x = 0  # TODO return a dictionary with the best images for each folder


find_best_image()
