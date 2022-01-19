"""A modification of create_lmdb_dataset.py in the case where gt label is the name of the image file"""

import fire
import os
import lmdb
import cv2

import numpy as np
from glob import glob
from create_lmdb_dataset import checkImageIsValid, writeCache


def createDataset(inputPath, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path        
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    
    datalist = glob(os.path.join(inputPath, "*.jpg"))  # create a list containing image file path
    labelList = list(map(lambda x: x.split('/')[-1].replace('.jpg', ''),datalist)) # create a list that contains the label corresponding filePath


    nSamples = len(datalist)
    for imagePath, label in zip(datalist, labelList):
        """
        imagePath label 예시
        /home/ubuntu/Dataset/text_recognition/Korean/public_crop/이해.jpg 이해
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
