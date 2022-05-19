import re, os
from typing import List, Tuple, Union



class CheckGT:
    def __init__(self, annotFile: str):
        
        self.annotFile = annotFile
        self.root_dir = annotFile.replace(annotFile.split('/')[-1], '')
        with open(annotFile, 'r') as f:
            annotTxt = f.readlines()
        for a in annotTxt:
            # 줄바꿈 문자만 있는 경우 제거
            # ['0 풀', '1 안내', '2 창피하다', '\n' ] -> ['0 풀', '1 안내', '2 창피하다' ]
            if a == '\n':
                annotTxt.remove(a)
        
        annotTxt = list(map(lambda x: x.replace('\n', ''), annotTxt))  # 줄바꿈 문자 제거 '0 풀\n' -> '0 풀'
        
        self.annotStrip = annotTxt
        self.Regex = re.compile('(\d)[ ](.+)')    # [숫자][공백][모든 문자열 하나 이상]
        self.img_annot = self.simple_pattern()
        self.file_annot = self.file_annot_pairing()
    
    def simple_pattern(self) -> Union[List[None], List[Tuple[str, str]]]:
        annot_check = []
        for txt in self.annotStrip:
            check = self.Regex.findall(txt)
            annot_check += check
        
        return annot_check
    
    
    def file_annot_pairing(self) -> Union[List[None], List[Tuple[str, str]]] :
        file_annot = []
        for prefix, annot in self.img_annot:
            fileName = os.path.join(self.root_dir, prefix+".jpg")
            if os.path.isfile(fileName):
                file_annot.append((fileName, annot))
            else:
                continue
        return file_annot
    
    
    def __lookup_pairs__(self, lk_len: int = 5) -> None:
        '''
        lk_len : length of sliced List
        '''
        print(self.file_annot[:lk_len])