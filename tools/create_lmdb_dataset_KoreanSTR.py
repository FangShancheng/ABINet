"""A modification of create_lmdb_dataset.py in the case where gt label is the name of the image file"""

import fire
import os
import lmdb
import cv2
import re, json

import numpy as np
import pandas as pd

from PIL import Image 
from matplotlib import pyplot as plt
from glob import glob
from create_lmdb_dataset import checkImageIsValid, writeCache
from create_lmdb_dataset_var import get_hangul, full2half

from typing import List, Dict, Tuple

root_path = '/Data2/Dataset/KoreanSTR/'

class KoreanSTRset:
    
    def __init__(self, labelDict):
        '''        
        ImageDict 
        e.g : [...{'id': '00000003',
             'width': 3739,
             'height': 175,
             'file_name': '00000003.png',
             'license': 'AI 오픈 이노베이션 허브'},
            ...], 
        
        AnnoDicts 
        e.g : [...{'id': '00000003',
                  'image_id': '00000003',
                  'text': '정권자였는데 이에 대한 사과가 우선"이라고 했다. 한편 문 후보와 김',
                  'attributes': {'type': '문장', 'gender': '여', 'age': '28', 'job': '직장인'}},
              ...], 
        '''
        labelDict: Dict[str, List]
        ImageDicts: List[Dict[str, str]] = labelDict['images']
        AnnoDicts: List[Dict[str, str]] = labelDict['annotations']            
        
        self.img_dir = {'글자(음절)': '1_syllable', '문장': '1_sentence', '단어(어절)': '1_word' }
        
        self.AnnotDF = pd.DataFrame(AnnoDicts)
        self.ImgDF = pd.DataFrame(ImageDicts)
        self.file_annot_pair = []
        del ImageDicts, AnnoDicts
        
        
    def imgView(self, filePath: str):
        '''
        filePath : complete path of single image file
        '''
        ImgArray = np.array(Image.open(filePath), dtype=np.uint8)
        plt.figure(figsize=(24, 12))
        ax1 = plt.subplot(111)
        ax1.imshow(ImgArray)
        plt.show()
        
    
    def imgName2path_annot(self, fileName: str, toView:bool = False) -> Tuple[str, str]:
        '''
        toView     matplotlib으로 이미지 시각화할 것인지 확인하는 옵션
        
        annot : pd.Series
        
        |  |   id     | image_id |                                  text                             |              attributes                 |
        | 1| 00000003 | 00000003 | 정권자였는데 이에 대한 사과가 우선"이라고 했다. 한편 문 후보와 김 | {'type': '문장', 'gender': '여',... }   |
        
        imgID : pd.Series
        
        |  |   id     |  width  | height |     file_name      |              license           |
        | 1| 00000003 |  3739   |  175   |    00000003.png    |     AI 오픈 이노베이션 허브    |
        
        '''
        imgID = self.ImgDF[self.ImgDF['file_name']==fileName]['id'].values[0]
        annot = self.AnnotDF[self.AnnotDF['id']==imgID]
        annot_text = annot['text'].values[0]
        annot_type = annot['attributes'].values[0]['type']
        imgFile = os.path.join(root_path + self.img_dir[annot_type], fileName)
        if toView:
            self.imgView(imgFile)
        return imgFile, annot_text
    
    
    def get_all_pairs(self) -> List[Tuple[str, str]]:
        
        total_imgs : List[str] # all the paths of available png files
        total_imgs_file : List[str]  # list of img file names ['00022234.png', '00000011.png', ..]
        img_annot_pairs : List[Tuple[str, str]] # list of image path - annotation pairs  [('/Data2/Dataset/KoreanSTR/1_syllable/00022234.png', '몜'), ...]
        
        total_imgs = []
        img_annot_pairs = []
        for folder in list(self.img_dir.values()):
            total_imgs += glob( root_path + folder + '/*.png')
            
        total_imgs_file = list(map(lambda x: x.split('/')[-1], total_imgs))
        del total_imgs
        
        img_annot_pairs += list(map(lambda imgQuery: self.imgName2path_annot(imgQuery), total_imgs_file[:10000]))  ## 실행되는지 확인하기 위해 100개만
        return img_annot_pairs
        




def createDataset(outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path        
        checkValid : if true, check the validity of every image
    """
    
    with open("/Data2/Dataset/KoreanSTR/handwriting_data_info1.json", "r", encoding='utf-8') as json_file:
        KorlabelDict = json.load(json_file)
    
    imgAnnot = KoreanSTRset(KorlabelDict)
    img_annot_pairs = imgAnnot.get_all_pairs()
    
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1


    nSamples = len(img_annot_pairs)
    for imagePath, label in img_annot_pairs:
        """
        imagePath label 예시 : '/Data2/Dataset/KoreanSTR/1_syllable/00022234.png' '몜'
        몜 -> ㅁㅖㅁ
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
