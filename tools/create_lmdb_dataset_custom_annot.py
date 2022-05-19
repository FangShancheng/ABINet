import fire
import os
import lmdb
import cv2
import re

import numpy as np
from glob import glob
from create_lmdb_dataset import checkImageIsValid, writeCache
from gt_file_check import CheckGT


def get_hangul(inputLabel):
    '''transform hangul character into 3 syllables
    Args :
        inputLabel : str
    Return :
        자모 분리해서 합친 뒤 공백(종성이 없는 경우)을 제거 : str
    '''
    base_unicode, initial_unicode, medium_unicode = 44032, 588, 28
    initial_list = list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')
    medium_list = list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
    terminal_list = list(' ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ') # 종성이 없는 경우도 있으므로 공백이 필요하다
    labelList = list(inputLabel)
    split_result = []
    for keyword in labelList:
        if re.match('.*[가-힣]+.*', keyword) is not None:
            # 초성
            char_code = ord(keyword) - base_unicode
            init_char = int(char_code/initial_unicode)
            split_result.append(initial_list[init_char])
            # 중성
            med_char = int((char_code - (initial_unicode*init_char))/medium_unicode)
            split_result.append(medium_list[med_char])
            # 종성
            term_char = int((char_code - (initial_unicode*init_char)-(medium_unicode*med_char)))
            split_result.append(terminal_list[term_char])
        else:
            split_result.append(keyword)
    output = "".join(split_result)
    return output.replace(' ', '')


def createDataset(gtPath, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : gt file path
        outputPath : LMDB output path        
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    
    syn_data = CheckGT(gtPath)

    nSamples = len(syn_data.file_annot)
    for imagePath, label in syn_data.file_annot:
        """
        imagePath label 예시 : '/Data1/FoodDetection/data/text_recognition/Korean/synthetic_data/data/0.jpg' '풀'
        풀 --> ㅍㅜㄹ
        """
    
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue
        label = get_hangul(label)

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode('utf-8')

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    fire.Fire(createDataset)