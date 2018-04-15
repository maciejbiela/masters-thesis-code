import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def crop(rect, img):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
               pts[1][0]:pts[2][0]]

    return img_crop


def srutututu(file):
    orig = cv2.imread(file)
    img = cv2.imread(file, 0)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    thresh = cv2.bitwise_not(thresh)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = None
    for contour in contours:
        if cv2.contourArea(contour) > 4000:
            rect = cv2.minAreaRect(contour)
    return crop(rect, orig)


def crop_all(numbers=['4', '5', '13', '14', '15', '16', '17']):
    for number in numbers:
        print(number)
        for file in os.listdir(number + '/raw'):
            if file.endswith('.png'):
                filename = number + '/raw' + '/' + file
                cropped = srutututu(filename)
                new_filename = filename.replace('/raw', '').replace('_000', '_')
                cv2.imwrite(new_filename, cropped)


crop_all(['4'])
