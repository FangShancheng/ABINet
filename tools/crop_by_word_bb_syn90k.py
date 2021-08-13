# Crop by word bounding box
# Locate script with gt.mat
# $ python crop_by_word_bb.py

import os
import re
import cv2
import scipy.io as sio
from itertools import chain
import numpy as np
import math

mat_contents = sio.loadmat('gt.mat')

image_names = mat_contents['imnames'][0]
cropped_indx = 0
start_img_indx = 0
gt_file = open('gt_oabc.txt', 'a')
err_file = open('err_oabc.txt', 'a')

for img_indx in range(start_img_indx, len(image_names)):


    # Get image name
    image_name_new = image_names[img_indx][0]
    # print(image_name_new)
    image_name = '/home/yxwang/pytorch/dataset/SynthText/img/'+ image_name_new
    # print('IMAGE : {}.{}'.format(img_indx, image_name))
    print('evaluating {} image'.format(img_indx), end='\r')
    # Get text in image
    txt = mat_contents['txt'][0][img_indx]
    txt = [re.split(' \n|\n |\n| ', t.strip()) for t in txt]
    txt = list(chain(*txt))
    txt = [t for t in txt if len(t) > 0 ]
    # print(txt) # ['Lines:', 'I', 'lost', 'Kevin', 'will', 'line', 'and', 'and', 'the', '(and', 'the', 'out', 'you', "don't", 'pkg']
    # assert 1<0

    # Open image
    #img = Image.open(image_name)
    img = cv2.imread(image_name, cv2.IMREAD_COLOR)
    img_height, img_width, _ = img.shape

    # Validation
    if len(np.shape(mat_contents['wordBB'][0][img_indx])) == 2:
        wordBBlen = 1
    else:
        wordBBlen = mat_contents['wordBB'][0][img_indx].shape[-1]

    if wordBBlen == len(txt):
        # Crop image and save
        for word_indx in range(len(txt)):
            # print('txt--',txt)
            txt_temp = txt[word_indx]
            len_now = len(txt_temp)
            # txt_temp = re.sub('[^0-9a-zA-Z]+', '', txt_temp)
            # print('txt_temp-1-',txt_temp)
            txt_temp = re.sub('[^a-zA-Z]+', '', txt_temp)
            # print('txt_temp-2-',txt_temp)
            if len_now - len(txt_temp) != 0:
                print('txt_temp-2-', txt_temp)

            if len(np.shape(mat_contents['wordBB'][0][img_indx])) == 2:  # only one word (2,4)
                wordBB = mat_contents['wordBB'][0][img_indx]
            else:  # many words (2,4,num_words)
                wordBB = mat_contents['wordBB'][0][img_indx][:, :, word_indx]

            if np.shape(wordBB) != (2, 4):
                err_log = 'malformed box index: {}\t{}\t{}\n'.format(image_name, txt[word_indx], wordBB)
                err_file.write(err_log)
                # print(err_log)
                continue

            pts1 = np.float32([[wordBB[0][0], wordBB[1][0]],
                               [wordBB[0][3], wordBB[1][3]],
                               [wordBB[0][1], wordBB[1][1]],
                               [wordBB[0][2], wordBB[1][2]]])
            height = math.sqrt((wordBB[0][0] - wordBB[0][3])**2 + (wordBB[1][0] - wordBB[1][3])**2)
            width = math.sqrt((wordBB[0][0] - wordBB[0][1])**2 + (wordBB[1][0] - wordBB[1][1])**2)

            # Coord validation check
            if (height * width) <= 0:
                err_log = 'empty file : {}\t{}\t{}\n'.format(image_name, txt[word_indx], wordBB)
                err_file.write(err_log)
                # print(err_log)
                continue
            elif (height * width) > (img_height * img_width):
                err_log = 'too big box : {}\t{}\t{}\n'.format(image_name, txt[word_indx], wordBB)
                err_file.write(err_log)
                # print(err_log)
                continue
            else:
                valid = True
                for i in range(2):
                    for j in range(4):
                        if wordBB[i][j] < 0 or wordBB[i][j] > img.shape[1 - i]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    err_log = 'invalid coord : {}\t{}\t{}\t{}\t{}\n'.format(
                        image_name, txt[word_indx], wordBB, (width, height), (img_width, img_height))
                    err_file.write(err_log)
                    # print(err_log)
                    continue

            pts2 = np.float32([[0, 0],
                               [0, height],
                               [width, 0],
                               [width, height]])

            x_min = np.int(round(min(wordBB[0][0], wordBB[0][1], wordBB[0][2], wordBB[0][3])))
            x_max = np.int(round(max(wordBB[0][0], wordBB[0][1], wordBB[0][2], wordBB[0][3])))
            y_min = np.int(round(min(wordBB[1][0], wordBB[1][1], wordBB[1][2], wordBB[1][3])))
            y_max = np.int(round(max(wordBB[1][0], wordBB[1][1], wordBB[1][2], wordBB[1][3])))
            # print(x_min, x_max, y_min, y_max)
            # print(img.shape)
            # assert 1<0
            if len(img.shape) == 3:
                img_cropped = img[ y_min:y_max:1, x_min:x_max:1, :]
            else:
                img_cropped = img[ y_min:y_max:1, x_min:x_max:1]
            dir_name = '/home/yxwang/pytorch/dataset/SynthText/cropped-oabc/{}'.format(image_name_new.split('/')[0])
            # print('dir_name--',dir_name)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            cropped_file_name = "{}/{}_{}_{}.jpg".format(dir_name, cropped_indx,
                                                         image_name.split('/')[-1][:-len('.jpg')], word_indx)
            # print('cropped_file_name--',cropped_file_name)
            # print('img_cropped--',img_cropped.shape)
            if img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
                err_log = 'word_box_mismatch : {}\t{}\t{}\n'.format(image_name, mat_contents['txt'][0][
                    img_indx], mat_contents['wordBB'][0][img_indx])
                err_file.write(err_log)
                # print(err_log)
                continue
            # print('img_cropped--',img_cropped)

            # img_cropped.save(cropped_file_name)
            cv2.imwrite(cropped_file_name, img_cropped)
            cropped_indx += 1
            gt_file.write('%s\t%s\n' % (cropped_file_name, txt[word_indx]))

            # if cropped_indx>10:
            #     assert 1<0
        # assert 1 < 0
    else:
        err_log = 'word_box_mismatch : {}\t{}\t{}\n'.format(image_name, mat_contents['txt'][0][
                                                            img_indx], mat_contents['wordBB'][0][img_indx])
        err_file.write(err_log)
        # print(err_log)
gt_file.close()
err_file.close()
