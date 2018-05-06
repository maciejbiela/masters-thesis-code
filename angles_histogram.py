import concurrent.futures
import itertools
import math
import os
from collections import Counter

import cv2
import numpy as np
from matplotlib import pyplot as plt
from color_histogram import compute_code, longest_code
from color_histogram import compare_hist


def equalize_light(color_image):
    yuv = cv2.cvtColor(color_image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(6, 6))
    y_equalized = clahe.apply(y)
    light_equalized = cv2.merge((y_equalized, u, v))
    return cv2.cvtColor(light_equalized, cv2.COLOR_YUV2BGR)


def normalize(hist):
    norm = np.linalg.norm(hist)
    if norm == 0:
        return hist
    return hist / norm


def angles_histogram(color_image):
    equalized = equalize_light(color_image)
    gray = cv2.cvtColor(equalized, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 300, 600)
    # plt.imshow(edges)
    # plt.show()
    lines = cv2.HoughLines(edges, 1, 1 / 180, 10)

    # In radians
    # angles = []
    # for line in lines:
    #     _, angle = line[0]
    #     angles.append(angle)
    #
    # angles = np.asarray(angles)
    # hist, buckets = np.histogram(angles, bins=180, range=(0, np.pi))

    # In degrees
    angles = []
    for line in lines:
        _, angle = line[0]
        angle = round(angle * 180 / np.pi)
        angles.append(angle)
    # print('no. of angles: {}'.format(len(angles)))
    angles = np.asarray(angles)

    hist, buckets = np.histogram(angles, bins=180, range=(0, 180))

    conv = np.convolve(hist, np.ones(6))
    print(np.argmax(conv))

    # plt.plot(buckets[:-1], hist)
    # plt.show()
    hist = hist.astype('float32')
    return normalize(hist)


def dominant_angle(color_image):
    equalized = equalize_light(color_image)
    gray = cv2.cvtColor(equalized, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 300, 600)
    # plt.imshow(edges)
    # plt.show()
    lines = cv2.HoughLines(edges, 1, 1 / 180, 10)

    # Hough Lines P
    angles = []
    lines_p = cv2.HoughLinesP(edges, 1, 1 / 180, 10, minLineLength=30, maxLineGap=5)

    for line in lines_p:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        else:
            a = (y1 - y2) / (x1 - x2)
        angle = np.arctan(a)
        angle *= 180 / np.pi
        if angle < 0:
            angle += 180
        angle = round(angle)
        if angle == 0:
            continue
        angles.append(angle)

    # In degrees
    # angles = []
    # for line in lines:
    #     _, angle = line[0]
    #     angle = angle * 180 / np.pi
    #     angle = round(angle * 2) / 2
    #     angles.append(angle)

    angles = np.asarray(angles)

    hist, _ = np.histogram(angles, bins=360, range=(0, 180))
    # plt.plot(range(0, 360), hist)
    # plt.show()

    convolution = np.convolve(hist, np.ones(6))
    return np.argmax(convolution) / 2


def rotate(color_image, angle, adjust_angle=False):
    if adjust_angle:
        angle -= 90

    height, width, _ = color_image.shape
    r = math.sqrt(width * width + height * height) / 2
    top = int(r - height / 2)
    side = int(r - width / 2)
    padded = cv2.copyMakeBorder(color_image, top, top, side, side, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    h, w, _ = padded.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(padded, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])


def compare_2(pair):
    first, second = pair
    splint1, file1, image1, angle1 = first
    splint2, file2, image2, angle2 = second
    # result = abs(angle1 - angle2)
    # if splint1 == splint2:
    #     if result <= threshold:
    #         outcomes.append('PP')
    #     else:
    #         print('FN: {}-{}: {}'.format(file1, file2, result))
    #         outcomes.append('FN')
    # else:
    #     if result <= threshold:
    #         print('FP: {}-{}: {}'.format(file1, file2, result))
    #         outcomes.append('FP')
    #     else:
    #         outcomes.append('NN')
    result = abs(angle1 - angle2)
    if result <= 10:
        result2 = compare_hist_result(image1, angle1, image2, angle2)
        print('Histogram comparison for {}:{} = {}'.format(file1, file2, result2))
        if result2 >= 0.6:
            if splint1 == splint2:
                return 'PP'
            else:
                return 'FP'
        else:
            if splint1 == splint2:
                return 'FN'
            else:
                return 'NN'
    else:
        if splint1 == splint2:
            return 'FN'
        else:
            return 'NN'


def compare(numbers=['4', '5', '6', '9', '10']):  # ,'13', '14', '15', '16', '17']):
    codes = []
    for number in numbers:
        folder = ''
        for file in os.listdir(folder + number):
            if file.endswith('.png'):
                filename = folder + number + '/' + file
                image = cv2.imread(filename)
                image = cv2.resize(image, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_CUBIC)
                # hist = angles_histogram(image)
                # codes.append((number, filename, hist))
                angle = dominant_angle(image)
                print('{}: {}'.format(file, angle))
                rotated = rotate(image, angle, True)
                plt.imshow(rotated)
                plt.show()
                plt.imshow(equalize_light(rotated))
                plt.show()
                codes.append((number, filename, image, angle))

    # outcomes = []
    # threshold = 0.8
    # combinations = itertools.combinations(codes, 2)
    # for (first, second) in combinations:
    #     splint1, file1, hist1 = first
    #     splint2, file2, hist2 = second
    #     result = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    #     print('{}-{}: {}'.format(file1, file2, result))
    #     if splint1 == splint2:
    #         if result > threshold:
    #             outcomes.append('PP')
    #         else:
    #             outcomes.append('FN')
    #     else:
    #         if result > threshold:
    #             outcomes.append('FP')
    #         else:
    #             outcomes.append('NN')
    #
    # print(Counter(outcomes))

    outcomes = []

    combinations = itertools.combinations(codes, 2)

    # for combination in combinations:
    #     outcomes.append(compare_2(combination))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for outcome in executor.map(compare_2, combinations):
            outcomes.append(outcome)

    print(Counter(outcomes))


def compare_hist_result(image1, angle1, image2, angle2):
    # return 0
    rotate1 = rotate(image1, angle1, True)
    rotate1 = equalize_light(rotate1)
    rotate2 = rotate(image2, angle2, True)
    rotate2 = equalize_light(rotate2)

    # plt.imshow(rotate1)
    # plt.show()
    # plt.imshow(rotate2)
    # plt.show()

    rotations = np.arange(-4, 4, 0.5)
    rotations1 = [rotate(rotate1, r) for r in rotations]
    max_result = -1
    for r1 in rotations1:
        # hist1 = normalize(compute_code(r1))
        # hist2 = normalize(compute_code(r2))
        hist1 = longest_code(r1)
        hist2 = longest_code(rotate2)
        max_result = max(max_result, compare_hist(hist1, hist2))
    # hist1 = longest_code(rotate1)
    # hist2 = longest_code(rotate2)
    # return compare_hist(hist1, hist2)
    return max_result

compare()
