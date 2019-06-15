# -*- coding: utf-8 -*-
# code based on https://docs.opencv.org/3.4.1/d1/de0/tutorial_py_feature_homography.html

import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt


MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1


def show_image(image):
    """
      Method that outputs an image

    Parameters
    ----------
    image : img
        The image to be shown
    """

    plt.imshow(image)
    plt.axis('off')
    plt.show()


def extract_document(query_img, train_img, out_mapping=False):
    """
     Method that, given two images (query and train) outputs the extracted query_img inside the train_img and,
     if chosen, outputs the mapping between the two images

    Parameters
    ----------
    query_img : str
        The path to the image of the identity document model
    train_img: str
        The path to the image that contains the identity document to detect
    out_mapping: bool
        Give in output also an image with the mapping between @query_img and @train_img. Default = False
    """

    img_query = cv.imread(query_img, 1)
    img_train = cv.imread(train_img, 1)

    sift = cv.xfeatures2d.SIFT_create()  # Initiate SIFT detector

    # find keypoints and descriptors with SIFT
    kp_qr, des_qr = sift.detectAndCompute(img_query, None)
    kp_tr, des_tr = sift.detectAndCompute(img_train, None)

    # use FLANN algorithms for a fast match between descriptors.
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_qr, des_tr, k=2)

    # store only good matches (Lowe's ratio test)
    goods = []
    for m, n in matches:
        if m.distance < 0.74 * n.distance:
            goods.append(m)

    print("Found %s matches" % (len(goods)))

    if len(goods) > MIN_MATCH_COUNT:  # the minimum accepted number of matches must be at least 4
        # extract the locations of matched keypoints
        src_pts = np.float32([kp_qr[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_tr[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)

        # compute the homography matrix
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        # see the link below for further details
        # https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html
        extracted_ID_img = cv.warpPerspective(img_train, np.linalg.inv(M), (img_query.shape[1], img_query.shape[0]))

        show_image(extracted_ID_img)

        if out_mapping == True:  # print the image with the mapping
            matchesMask = mask.ravel().tolist()

            # do the perspective transformation and find the object
            h, w, d = img_query.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)

            # draw on img2 the object borders
            segmented_img = cv.polylines(img_train, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

            # draw a properly formatted image with the matches found
            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
            mapping_img = cv.drawMatches(img_query, kp_qr, segmented_img, kp_tr, goods, None, **draw_params)

            show_image(mapping_img)

    else:
        print("Not enough matches are found")


im1 = "/home/vale/Documenti/Project Course/Immagini/Match documenti/model_30_ita.png"
im2 = "/home/vale/Documenti/Project Course/Immagini/Match documenti/ita_drvid_2.jpg"
extract_document(im1, im2, True)

# if len(sys.argv) == 3:
#     ExtractDocument(sys.argv[1],sys.argv[2])
# elif len(sys.argv) == 4:
#     bool_argv3 = sys.argv[3].lower() == 'true'
#     ExtractDocument(sys.argv[1], sys.argv[2], bool_argv3)
# else:
#     print("Wrong parameters number")
