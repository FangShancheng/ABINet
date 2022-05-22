import fire
import os
import lmdb
import cv2
import re

import numpy as np
from glob import glob
from create_lmdb_dataset import checkImageIsValid, writeCache
from gt_file_check import CheckGT
from create_lmdb_dataset_var import get_hangul, full2half



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
        
        label = full2half(label)
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