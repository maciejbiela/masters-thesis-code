import itertools
import os
from collections import Counter

import cv2
import numpy as np
from matplotlib import pyplot as plt


def select_sample(grayscale, step=50, threshold=100):
    mask = grayscale.copy()
    with_nans = grayscale.copy().astype('float32')
    with_nans[mask >= threshold] = np.nan
    height, _ = grayscale.shape
    selected_rows = []
    for row_number in range(0, height, step):
        selected_rows.append(with_nans[row_number])
    return selected_rows


def drop_nans(h):
    return h[np.isfinite(h)]


def compare_hist(h1, h2):
    # rotated_h1 = np.fliplr([h1])[0]
    regular = compare_hist_one_way(h1, h2)
    # rotated = compare_hist_one_way(rotated_h1, h2)
    # return max(regular, rotated)
    return regular

def compare_hist_one_way(h1, h2):
    h1 = h1.astype('float32')
    h2 = h2.astype('float32')
    l1 = np.size(h1)
    l2 = np.size(h2)
    if l1 == l2:
        # return np.correlate(h1, h2)
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    elif l1 > l2:
        return find_max_similarity(h1, h2, l1, l2)
    else:
        return find_max_similarity(h2, h1, l2, l1)


def find_max_similarity(longer, shorter, l_longer, l_shorter):
    diff = l_longer - l_shorter
    max_comparison_result = 0
    for offset in range(diff):
        comparison_result = compare_hist_one_way(longer[offset:offset + l_shorter], shorter)
        max_comparison_result = max(max_comparison_result, comparison_result)
    return max_comparison_result


#
# file1 = 'rotated2.jpg'
# file2 = 'rotated3.jpg'
# gray1 = cv2.imread(file1, 0)
# gray2 = cv2.imread(file2, 0)
#
# rows1 = select_sample(gray1)
# rows2 = select_sample(gray2)
#
# median1 = extract_the_meat(np.nanmedian(rows1, axis=0))
# median2 = extract_the_meat(np.nanmedian(rows2, axis=0))
#
# _, axarr = plt.subplots(nrows=2, ncols=1, sharex='col')
# axarr[0].plot(range(0, np.size(median1)), median1)
# axarr[1].plot(range(0, np.size(median2)), median2)
#
# plt.show()

def do_nothing(color_image):
    return color_image


def equalize_light(color_image):
    yuv = cv2.cvtColor(color_image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    # clahe = cv2.createCLAHE(clipLimit=2.0)
    clahe = cv2.createCLAHE(clipLimit=80.0, tileGridSize=(6, 6))
    y_equalized = clahe.apply(y)
    # y_equalized = cv2.equalizeHist(y)
    light_equalized = cv2.merge((y_equalized, u, v))
    return cv2.cvtColor(light_equalized, cv2.COLOR_YUV2BGR)


def bilateral_filtered(color_image):
    return cv2.bilateralFilter(color_image, 9, 75, 75)


def convert_to_grayscale(color_image):
    preprocessed = do_nothing(color_image)
    return cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)


def compare(numbers=['13', '14', '15', '16', '17']):
    codes = []
    for number in numbers:
        for file in os.listdir(number):
            if file.endswith('.png'):
                filename = number + '/' + file
                color_image = cv2.imread(filename)
                hist = compute_code(color_image)
                # diff = derivative(hist)
                codes.append((number, filename, normalize(hist)))

                # code = longest_code(sample)
                # codes.append((number, code))

                # print('{}, {}'.format(diff, len(diff)))

    outcomes = []
    threshold = 0.5
    combinations = itertools.combinations(codes, 2)
    for (first, second) in combinations:
        splint1, file1, hist1 = first
        splint2, file2, hist2 = second
        result = compare_hist(hist1, hist2)
        print('{}-{}: {}'.format(file1, file2, result))
        if splint1 == splint2:
            if result > threshold:
                outcomes.append('PP')
            else:
                outcomes.append('FN')
        else:
            if result > threshold:
                outcomes.append('FP')
            else:
                outcomes.append('NN')

    print(Counter(outcomes))


def compute_code(color_image):
    grayscale = convert_to_grayscale(color_image)
    sample = select_sample(grayscale, step=20)
    # sample = select_sample(image)
    hist = histogram(sample)
    return hist


def normalize(h):
    norm = np.linalg.norm(h)
    if norm == 0:
        return h
    return h / norm


def histogram(sample):
    statistic = np.nanmean(sample, axis=0)
    return drop_nans(statistic)


def derivative(hist):
    return np.diff(hist)


# Experimental
def longest_code(color_image):

    grayscale = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    mask = grayscale.copy()
    with_nans = grayscale.copy().astype('float32')
    with_nans[mask >= 200] = np.nan


    index = -1
    max_index = 0
    max_length = 0
    for row in with_nans:
        index += 1
        length = np.count_nonzero(~np.isnan(row))
        if length > max_length:
            max_length = length
            max_index = index

    rows = with_nans[max_index - 10: max_index + 10]
    return normalize(histogram(rows))


# compare()


# def close_to_zero(deriv, thresh):
#     indexes = []
#     for index in range(np.size(deriv)):
#         if -thresh <= deriv[index] <= thresh:
#             indexes.append(index)
#
#     return indexes
#
#
# def rapid_change(filename):
#     img = cv2.imread(filename)
#     filtered = cv2.bilateralFilter(img, 9, 75, 75)
#     gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
#     sample = select_sample(gray)
#     hist = histogram(sample)
#     diff = derivative(hist)
#     indexes = close_to_zero(diff, 0.1)
#     # for index in indexes:
#     #     hist[index] = 255
#     return np.outer(np.ones(300), hist)
#
#

def longest_diff(param):
    gray = cv2.imread(param, 0)
    sample = select_sample(gray, step=17)
    hist = histogram(sample)
    return normalize(hist)


def meh():
    rc_15_04 = longest_diff('Square-Code.jpg')
    rc_15_11 = longest_diff('Square2-Code.jpg')
    _, axarr = plt.subplots(nrows=2, ncols=1, sharex='col')
    axarr[0].plot(range(np.size(rc_15_04)), rc_15_04)
    axarr[1].plot(range(np.size(rc_15_11)), rc_15_11)
    plt.show()
    print(compare_hist(rc_15_04, rc_15_11))

# TODO:
# 1. First bucket it up according to angles
