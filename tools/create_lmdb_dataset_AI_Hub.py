"""A modification of create_lmdb_dataset.py in the case where gt label is the name of the image file"""

import fire
import os
import lmdb

import re
import json

from glob import glob
from create_lmdb_dataset import checkImageIsValid, writeCache
from create_lmdb_dataset_var import get_hangul

from typing import List, Dict
from functools import singledispatch


@singledispatch
def get_label(arg):
    #return arg
    raise NotImplementedError(f"Cannot format value of type {type(arg)}")

@get_label.register(dict)
def valueType(arg: Dict) -> str:
    return arg['value']

@get_label.register(list)
def valueType(arg: List) -> str:
    return ''.join([lb['value'] for lb in arg])


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
    
    datalist = glob(os.path.join(inputPath, "*/*.jpg"))    # create a list containing image file path
    labelList = glob(os.path.join(inputPath, "*/*.json")) # create a list that contains the label corresponding filePath
    
    nSamples = len(datalist)
    for imagePath, labelPath in zip(datalist, labelList):
        """
        label JSON 예시
        {'info': {'name': 'Korean OCR Data Set', 'description': 'Korean OCR Data Set (letter handwrite)', 'date_created': '2020-12-22 13:39:21', 'text': '각'}, 'image': {'file_name': '00130001002.jpg', 'width': 110, 'height': 110, 'dpi': 300, 'bit': 24}, 'text': {'type': 'letter', 'output': 'handwrite', 'letter': {'value': '각'}}, 'license': {'output': 'handwrite', 'font': '', 'font_no': '', 'font_license': '', 'font_url': '', 'writer_no': '001', 'writer_gender': 'female', 'writer_age': '40'}}
        """
    
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            
        with open(labelPath, "r") as lbl:
            lbl_dict = json.load(lbl)
        
        try:
            if not lbl_dict['image']['file_name'] == imagePath.split('/')[-1] :
                print(lbl_dict['image']['file_name'], " does not match the corresponding image file name ", imagePath.split('/')[-1])
                continue
                
        except KeyError as ke:
            print(ke)
            continue
            
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
        
        label = get_label(lbl_dict['text'][lbl_dict['text']['type']])
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
