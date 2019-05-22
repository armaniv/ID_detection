# -*- coding: utf-8 -*-
# code taken from https://docs.opencv.org/3.4.1/d1/de0/tutorial_py_feature_homography.html

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

PATH = '/media/vale/HDDVale/Uni/DataSet'
MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1

img1 = cv.imread(PATH + '/30_ita_drvlic/images/30_ita_drvlic.tif', 1)     # queryImage
img2 = cv.imread(PATH + '/30_ita_drvlic/images/CA/CA30_02.tif', 1)             # trainImage

sift = cv.xfeatures2d.SIFT_create()                                     # Initiate SIFT detector

# find keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# use FLANN algorithms for a fast match between descriptors.
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store only good matches (Lowe's ratio test)
goods = []
for m, n in matches:
    if m.distance < 0.74 * n.distance:
        goods.append(m)

print("Found %s matches" % (len(goods)))

if len(goods) > MIN_MATCH_COUNT:                                         # the minimum accepted number of matches is 4
    # extract the locations of matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)

    # find the perspective transformation
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # find the object
    h, w, d = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)

    # draw on img2 the object borders
    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

else:
    matchesMask = None

# draw a properly formatted image with the matches found
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, goods, None, **draw_params)

plt.imshow(img3)
plt.axis('off')
plt.show()
