""" Created by MrBBS """
# 10/3/2022
# -*-encoding:utf-8-*-

import os
import random
import numpy as np

import pandas as pd
from tqdm import tqdm

max_length = 50
min_length = 2
root = '../data'
charset = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸėêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ'
digits = '0123456789'
chars = ''.join(open(r'D:\SourceCode\text_recognize\DAN\char.txt', 'r', encoding='utf-8').read().strip().split('\n'))


def is_char(text, ratio=0.5):
    # text = text.lower()
    length = max(len(text), 1)
    char_num = sum([t in charset for t in text])
    if char_num < min_length: return False
    if char_num / length < ratio: return False
    return True


def is_digit(text, ratio=0.5):
    length = max(len(text), 1)
    digit_num = sum([t in digits for t in text])
    if digit_num / length < ratio: return False
    return True


lines = list(set(open(r'E:\NLP_data\words_VN.txt', 'r', encoding='utf-8').read().strip().split('\n') +
                 open(r'E:\NLP_data\words_english.txt', 'r', encoding='utf-8').read().strip().split('\n')))
np.random.shuffle(lines)
inp, gt = [], []
for line in tqdm(lines, desc='Create train set'):
    # token = line.split()
    # for text in token:
    # text = re.sub('[^0-9a-zA-Z]+', '', text)
    if len(line) < min_length:
        # print('short-text', text)
        continue
    if len(line) > max_length:
        # print('long-text', text)
        continue
    if not set(line).issubset(chars):
        continue
    inp.append(line)
    gt.append(line)
del lines
train_voc = os.path.join(root, 'VN_ENG.csv')
pd.DataFrame({'inp': inp, 'gt': gt}).to_csv(train_voc, index=None, sep='\t')
del gt


def disturb(word, degree, p=0.3):
    if len(word) // 2 < degree: return word
    if is_digit(word): return word
    if random.random() < p:
        return word
    else:
        index = list(range(len(word)))
        random.shuffle(index)
        index = index[:degree]
        new_word = []
        for i in range(len(word)):
            if i not in index:
                new_word.append(word[i])
                continue
            if (word[i] not in charset) and (word[i] not in digits):
                # special token
                new_word.append(word[i])
                continue
            op = random.random()
            if op < 0.1:  # add
                new_word.append(random.choice(charset))
                new_word.append(word[i])
            elif op < 0.2:
                continue  # remove
            else:
                new_word.append(random.choice(charset))  # replace
        return ''.join(new_word)


lines = inp
degree = 1
keep_num = 50000

np.random.shuffle(lines)
part_lines = lines[:keep_num]
del lines
inp, gt = [], []

for w in tqdm(part_lines, desc='Create val set'):
    # token = w.split()
    # for text in token:
    #     text = text.strip()
    new_w = disturb(w, degree)
    inp.append(new_w)
    gt.append(w)

eval_voc = os.path.join(root, f'VN_ENG_eval{degree}.csv')
pd.DataFrame({'inp': inp, 'gt': gt}).to_csv(eval_voc, index=None, sep='\t')
