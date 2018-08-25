import os

import cv2
import numpy as np


def crop(rect, img):
    # rotate img
    angle = -rect[2]

    M, nW, nH = rotate_parameters(img, angle)
    img_rot = cv2.warpAffine(img, M, (nW, nH), borderValue=(255, 255, 255))

    # rotate bounding box
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
               pts[1][0]:pts[2][0]]

    return img_crop


def rotate_parameters(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return M, nW, nH


def rotate_bound(image, angle):
    M, nW, nH = rotate_parameters(image, angle)

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))


def rotate_if_necessary(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    up = 0
    down = 0
    left = 0
    right = 0
    shift = 100
    thresh = 200
    for i in range(width):
        if gray[shift][i] < thresh:
            up += 1
        if gray[height - shift][i] < thresh:
            down += 1

    for i in range(height):
        if gray[i][shift] < thresh:
            left += 1
        if gray[i][width - shift] < thresh:
            right += 1

    if max(up, down, left, right) == down:
        return cropped
    elif max(up, down, left, right) == up:
        return rotate_bound(cropped, 180)
    elif max(up, down, left, right) == left:
        return rotate_bound(cropped, 270)
    else:
        return rotate_bound(cropped, 90)


def extract_boxed_splint(file):
    orig = cv2.imread(file)
    img = cv2.imread(file, 0)
    ret, thresh = cv2.threshold(img, 200, 255, 0)
    thresh = cv2.bitwise_not(thresh)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = None
    for contour in contours:
        if cv2.contourArea(contour) > 150000:
            rect = cv2.minAreaRect(contour)
    if rect is None:
        raise Exception
    cropped = crop(rect, orig)
    return rotate_if_necessary(cropped)


def crop_all(folder, numbers=['4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']):
    for number in numbers:
        for file in os.listdir(folder + number + '/raw'):
            if file.endswith('.png'):
                try:
                    filename = folder + number + '/raw' + '/' + file
                    cropped = extract_boxed_splint(filename)
                    new_filename = filename.replace('/raw', '').replace('_000', '_')
                    print('Rotating {}'.format(new_filename))
                    if not os.path.isfile(new_filename):
                        cv2.imwrite(new_filename, cropped)
                except:
                    print('Skipping {}'.format(filename))
