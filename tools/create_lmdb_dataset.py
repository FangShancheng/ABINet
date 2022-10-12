""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import os
import lmdb
import cv2

import numpy as np
from tqdm import tqdm


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True, map_size=5073741824):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=map_size)
    cache = {}
    cnt = 1

    datalist = open(gtFile, 'r', encoding='utf-8').read().strip().split('\n')

    print(len(datalist))
    for i, sample in tqdm(enumerate(datalist)):
        try:
            imagePath, label = sample.split('\t')
            if len(label) < 51:
                imagePath = os.path.join(inputPath, imagePath)

                # # only use alphanumeric data
                # if re.search('[^a-zA-Z0-9]', label):
                #     continue

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
                cache[labelKey] = label.strip().encode()

                if cnt % 1000 == 0:
                    writeCache(env, cache)
                    cache = {}
                    print('Written %d / %d' % (cnt, i))
                cnt += 1
        except Exception as e:
            print(sample, e)
    i = cnt - 1
    cache['num-samples'.encode()] = str(i).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % i)


if __name__ == '__main__':
    # createDataset(r'E:\text_recognize_data\syntext_word', r'E:\text_recognize_data\syntext_word\real_synth.txt', r'E:\text_recognize_data\syntext_word\lmdb')
    createDataset(r'E:\text_recognize_data\syntext_line\synth', r'E:\text_recognize_data\syntext_line\synth\gt.txt',
                  r'E:\text_recognize_data\syntext_line\lmdb', map_size=5073741824)
    createDataset(r'E:\text_recognize_data\syntext_line\synth_test',
                  r'E:\text_recognize_data\syntext_line\synth_test\gt.txt',
                  r'E:\text_recognize_data\syntext_line\lmdb_val', map_size=1073741824)

    # createDataset(r'E:\text_recognize_data\syntext_word\synth_test',
    #               r'E:\text_recognize_data\syntext_word\synth_test\gt.txt',
    #               r'E:\text_recognize_data\syntext_word\lmdb_val')
