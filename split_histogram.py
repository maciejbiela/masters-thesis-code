import itertools
import os
from datetime import datetime
from statistics import mean

import cv2
import numpy as np

from color_histogram import compare_hist, longest_code
from rotate import rotate_bound, crop_all


def equalize_light(color_image):
    yuv = cv2.cvtColor(color_image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(6, 6))
    y_equalized = clahe.apply(y)
    light_equalized = cv2.merge((y_equalized, u, v))
    return cv2.cvtColor(light_equalized, cv2.COLOR_YUV2BGR)


def dominant_angle(color_image):
    equalized = equalize_light(color_image)
    gray = cv2.cvtColor(equalized, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 300, 600)

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
        angle = round(angle * 2) / 2
        if angle == 0:
            continue
        angles.append(angle)

    angles = np.asarray(angles)

    hist, _ = np.histogram(angles, bins=360, range=(0, 180))
    convolution = np.convolve(hist, np.ones(6))
    return np.argmax(convolution) / 2.0


def code(color_image):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    height, _ = gray.shape
    half_height = int(height / 2)
    single_row = gray[half_height]

    hist = np.asarray([v for v in single_row if v < 200])
    norm = np.linalg.norm(hist)
    if norm == 0:
        return hist
    return hist / norm


def compare(image1, image2):
    height_1, _, _ = image1.shape
    height_2, _, _ = image2.shape
    half_height_1 = int(height_1 / 2)
    half_height_2 = int(height_2 / 2)
    image1_up = image1[:half_height_1]
    image2_up = image2[:half_height_2]
    rotated1_up, angle1 = rotate(image1_up)
    rotated2_up, angle2 = rotate(image2_up)

    # plt.imshow(image1_up)
    # plt.show()
    #
    # plt.imshow(rotated1_up)
    # plt.show()
    #
    # hist = longest_code(rotated1_up)
    # hist2 = np.outer(np.ones(100), hist)
    #
    # plt.imshow(hist2, cmap='gray')
    # plt.show()
    #
    # plt.plot(range(len(hist)), hist)
    # plt.show()

    if abs(angle1 - angle2) > 5:
        return -1

    rotations = np.arange(-3, 3, 0.5)
    rotations1 = [rotate_bound(rotated1_up, r) for r in rotations]
    max_result = -1
    for r1 in rotations1:
        hist1 = longest_code(r1)
        hist2 = longest_code(rotated2_up)
        max_result = max(max_result, compare_hist(hist1, hist2))

    return max_result


def rotate(image):
    angle = dominant_angle(image) - 90
    return rotate_bound(image, -angle), angle


def print_statistics(output, results):
    non_null_results = [x for x in results if x != -1]
    if len(non_null_results) > 0:
        print('{}: mean={}'.format(output, mean(non_null_results)))


def results_for_angle(caption, pp, fp, nn, fn):
    return """
\\begin{{table}}[H]
    \\centering
    \\begin{{spacing}}{{1.5}}    
    \\begin{{tabular}}{{|l|l|l|}}
        \\hline
        \\cellcolor{{gray}} & \\textbf{{Output: Same}} & \\textbf{{Output: Different}} \\\\ [0.5ex]
        \\hline\\hline
        \\textbf{{Actual: Same}} & {pp} & {fn} \\\\ [0.5ex]
        \\hline
        \\textbf{{Actual: Different}} & {fp} & {nn} \\\\ [0.5ex]
        \\hline
    \\end{{tabular}}
    \\caption{{{caption}}}
    \\end{{spacing}}
\\end{{table}}
        """.format(caption=caption, pp=len(pp), fn=len(fn), fp=len(fp), nn=len(nn))


def total_statistics(caption, tpp, tfp, tnn, tfn):
    output = """
\\begin{{table}}[H]
    \\begin{{spacing}}{{1.5}}
    \\centering
    \\begin{{tabular}}{{|l|l|l|}}
        \\hline
        \\cellcolor{{gray}} & \\textbf{{Output: Same}} & \\textbf{{Output: Different}} \\\\ [0.5ex]
        \\hline\\hline
        \\textbf{{Actual: Same}} & {pp} & {fn} \\\\ [0.5ex]
        \\hline
        \\textbf{{Actual: Different}} & {fp} & {nn} \\\\ [0.5ex]
        \\hline
    \\end{{tabular}}
    \\caption{{{caption}}}
    \\end{{spacing}}
\\end{{table}}

        """.format(caption=caption, pp=tpp, fn=tfn, fp=tfp, nn=tnn)
    tpr = round(tpp / (tpp + tfn), 2)
    spc = round(tnn / (tnn + tfp), 2)
    ppv = round(tpp / (tpp + tfp), 2)
    npv = round(tnn / (tnn + tfn), 2)
    fpr = round(1 - spc, 2)
    fnr = round(1 - tpr, 2)
    fdr = round(1 - ppv, 2)
    acc = round((tpp + tnn) / (tpp + tfp + tfn + tnn), 2)
    f1 = round(2 * tpp / (2 * tpp + tfp + tfn), 2)
    output += f"""
Sensitivity                 = {tpr}
Specificity                 = {spc}
Positive predictive value   = {ppv}
Negative predictive value   = {npv}
False positive rate         = {fpr}
False negative rate         = {fnr}
False discovery rate        = {fdr}
Accuracy                    = {acc}
F1 score                    = {f1}        
    """
    return output


def run_algorithm(folders, numbers=['4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']):
    output = ''

    tpp = 0
    tfp = 0
    tnn = 0
    tfn = 0

    for folder in folders:

        codes = []

        for number in numbers:

            try:
                for file in os.listdir(folder + number):
                    if file.endswith('.png'):
                        filename = folder + number + '/' + file
                        image = cv2.imread(filename)
                        image = cv2.resize(image, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_CUBIC)
                        codes.append((number, filename, equalize_light(image)))
                        # codes.append((number, filename, image))
            except:
                abs(10)

        pp = []
        fp = []
        nn = []
        fn = []
        match = []
        mismatch = []

        outcomes = []
        combinations = itertools.combinations(codes, 2)
        for (first, second) in combinations:
            splint1, file1, image1 = first
            splint2, file2, image2 = second
            result = compare(image1, image2)
            print('Comparing {}-{}: {}'.format(file1, file2, result))
            thresh = .635
            if splint1 == splint2:
                append_result(match, result)
                if result > thresh:
                    outcomes.append('PP')
                    append_result(pp, result)
                else:
                    outcomes.append('FN')
                    append_result(fn, result)
            else:
                append_result(mismatch, result)
                if result > thresh:
                    outcomes.append('FP')
                    append_result(fp, result)
                else:
                    outcomes.append('NN')
                    append_result(nn, result)

        # print(Counter(outcomes))
        # print_statistics('PP', pp)
        # print_statistics('FP', fp)
        # print_statistics('NN', nn)
        # print_statistics('FN', fn)
        # print_statistics('Match', match)
        # print_statistics('Mismatch', mismatch)

        if (folder.startswith('deg_')):
            output += results_for_angle(folder.replace('deg_', 'Results for angle ').replace('/', '$^{\circ}$'), pp, fp,
                                        nn, fn) + '\n'

        tpp += len(pp)
        tfp += len(fp)
        tnn += len(nn)
        tfn += len(fn)

    if len(folders) > 1:
        output += total_statistics('Aggregated results', tpp, tfp, tnn, tfn)
    else:
        output += total_statistics('Overall results', tpp, tfp, tnn, tfn)

    return output


def append_result(list, result):
    list.append(result)


def perform_comparisons(folders):
    for folder in folders:
        crop_all(folder)
    return run_algorithm(folders)

print(datetime.now())

print(perform_comparisons([
    'deg_0/',
    'deg_45/',
    'deg_90/',
    'deg_135/',
    'deg_180/',
    'deg_225/',
    'deg_270/',
    'deg_315/'
]))

# print(perform_comparisons(['all/']))

print(datetime.now())
