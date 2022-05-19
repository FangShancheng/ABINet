# 데이터 전처리  
---  
## MJSynth
ABINet은 LDMB 형식의 데이터를 학습하기 때문에 Raw data를 이에 맞게 변형시켜주어야 한다.<br>
기본적으로 제공되는 전처리 스크립트인 `create_lmdb_dataset.py`는 MJSynth 처럼 별도의 annotation 파일이 있고, 그 파일에 다음과 같이 이미지 경로와 라벨이 적혀 있는 방식이다. 이미지 경로와 라벨은 공백으로 구분될 수도 있고, 탭으로 구분될 수도 있으므로 확인 후 파일을 실행시키면 된다.<br>

```
./3000/7/181_REMODELERS_64541.jpg 64541
./3000/7/180_Chronographs_13538.jpg 13538
./3000/7/179_Impeaching_38222.jpg 38222
./3000/7/178_discombobulated_22063.jpg 22063
```
<br>
문제는 annotation file이 위와 같은 형식인데, label 부분이 문자열이 아니고 숫자로 되어 있음을 알 수 있다. 오히려 정답 문자열은 파일이름에 들어 있다. 따라서 이 경우는 파일 이름에서 label을 추출하는 코드를 추가한 <code>create_lmdb_dataset_MJ.py</code>를 사용해서 데이터셋을 만들어 주어야 한다.
<br>  

```bash
python tools/create_lmdb_dataset_MJ.py --inputPath=/home/ubuntu/Dataset/text_recognition/Korean/mjsynth --outputPath=/home/ubuntu/Playground/ABINet/data/training/MJ/MJ_test --gtFile=/home/ubuntu/Dataset/text_recognition/Korean/mjsynth/annotation_test.txt  
```
<br>  

---------------------  
## Custom annotation file
Annotation 파일의 형식이 다음과 같이 `확장자를 제외한 파일명-공백-GT 문자열` 형태로 되어 있는 경우.

```
0 풀
1 안내
2 창피하다
3 소리치다
4 칠십
5 무섭다
6 진행자
7 서다
8 구르다
9 너
```

특히 GT 문자열에 특수문자나 공백이 포함되어 있는 경우에는

```
0 미나미오미네 역 ()은 일본 야마
1 이런멋진신인을어디서!!
2 대학교 동문 분류:시드니 출신
3 르겐 출신 분류:오슬로 대학교 동문
4 2편은힛걸에의한힛걸을위한힛
5 ellochloris 분류:녹조강
6  분류:대한민국의 교육공무원
7 nd free. 국가 분류:국가
8 이츠하크 샤미르 (, Yitzha
9  배우 분류:서울특별시 출신
```

파일명과 GT를 분리할 때 단순히 공백으로 자르면 안되고, 정규표현식 사용해서 분리해야 한다.

gt_file_check.py 파일 중 `CheckGT` 클래스에 다음과 같은 정규표현식을 적용해서 튜플로 파일명과 GT 문자열을 분리하였다.

```python
class CheckGT:
    def __init__(self, annotFile: str):        

        self.Regex = re.compile('(\d)[ ](.+)')    # [숫자][공백][모든 문자열 하나 이상]
    
    def simple_pattern(self) -> Union[List[None], List[Tuple[str, str]]]:
        annot_check = []
        for txt in self.annotStrip:
            check = self.Regex.findall(txt)
            annot_check += check
        
        return annot_check
  
```

메소드 `simple_pattern`를 실행하면 

```python
[('0', '미나미오미네 역 ()은 일본 야마'),
 ('1', '이런멋진신인을어디서!!'),
 ('2', '대학교 동문 분류:시드니 출신'),
 ('3', '르겐 출신 분류:오슬로 대학교 동문'),
 ('4', '2편은힛걸에의한힛걸을위한힛'),
 ('5', 'ellochloris 분류:녹조강'),
 ('6', ' 분류:대한민국의 교육공무원'),
 ('7', 'nd free. 국가 분류:국가'),
 ('8', '이츠하크 샤미르 (, Yitzha'),
 ('9', ' 배우 분류:서울특별시 출신'),
...]
```

처럼 리턴된다.

메소드 `file_annot_pairing`를 실행하면 튜플의 첫번째 요소인 파일명 뒤에 확장자를 concat하고 파일명 앞에는 폴더경로를 concat해서 완전한 파일 경로로 바꿔준다. 그러면 lmdb 형태로 바꿀 준비가 된것이다.

```python
[('/Data1/FoodDetection/data/text_recognition/Korean/synthetic_data/hardcore1/0.jpg',
  '미나미오미네 역 ()은 일본 야마'),
 ('/Data1/FoodDetection/data/text_recognition/Korean/synthetic_data/hardcore1/1.jpg',
  '이런멋진신인을어디서!!'),
 ('/Data1/FoodDetection/data/text_recognition/Korean/synthetic_data/hardcore1/2.jpg',
  '대학교 동문 분류:시드니 출신'),
 ('/Data1/FoodDetection/data/text_recognition/Korean/synthetic_data/hardcore1/3.jpg',
  '르겐 출신 분류:오슬로 대학교 동문'),
 ('/Data1/FoodDetection/data/text_recognition/Korean/synthetic_data/hardcore1/4.jpg',
  '2편은힛걸에의한힛걸을위한힛'), ...]
```

lmdb 변환은 다음과 같이 `gtPath`와 `outputPath` 인자를 주면 문제없이 수행된다.

```bash
$ python create_lmdb_dataset_custom_annot.py --gtPath=/Data1/FoodDetection/data/text_recognition/Korean/synthetic_data/data/gt.txt --outputPath=/Data2/Dataset/Preproc/ABINet/synthetic_data/
$ python create_lmdb_dataset_custom_annot.py --gtPath=/Data1/FoodDetection/data/text_recognition/Korean/synthetic_data/data_long4/gt_long.txt --outputPath=/Data2/Dataset/Preproc/ABINet/data_long4/
$ python create_lmdb_dataset_custom_annot.py --gtPath=/Data1/FoodDetection/data/text_recognition/Korean/synthetic_data/hardcore1/gt.txt --outputPath=/Data2/Dataset/Preproc/ABINet/hardcore1/
```



-----

## When Ground Truth is a File name itself
반면 별도의 annotation 파일은 없고, 이미지 파일이름을 라벨로 하는 데이터셋도 있다.<br>   

```
/home/ubuntu/Dataset/custom/라면.jpg  --> 라면
```
<br>
이 경우는 <code>create_lmdb_dataset_var.py</code>을 실행시키면 된다.
<br>
<br>

--------------------  

## Annotation as a JSON File 1  
Annotation이 JSON 형식인 경우에도, 적용할 수 있다. 대표적으로 AI Hub의 <a href=https://aihub.or.kr/aidata/33987> 한글 손글씨 데이터셋 </a>이 있다.<br>
이 데이터셋은 기본적으로 글자단위 Annotation이 되어 있고, 단어의 경우에는 글자 Annotation의 리스트 형식이다.<br>
바로 아래에 나오는 JSON이 글자 하나의 Annotation이다. <code>text</code>의 <code>type</code>이 <code>"letter"</code>이고 <code>letter</code>의 값은 dictionary <code>{'value': '가'}</code>이다.<br>



```json
{"info": {"name": "Korean OCR Data Set",
  "description": "Korean OCR Data Set (letter handwrite)",
  "date_created": "2020-12-22 13:38:11",
  "text": "가"},
 "image": {"file_name": "00130001001.jpg",
  "width": 111,
  "height": 110,
  "dpi": 300,
  "bit": 24},
 "text": {"type": "letter", "output": "handwrite", "letter": {"value": "가"}},
 "license": {"output": "handwrite",
  "font": "",
  "font_no": "",
  "font_license": "",
  "font_url": "",
  "writer_no": "001",
  "writer_gender": "female",
  "writer_age": "40"}}
```  

그 다음에 나오는 JSON이 단어의 Annotation이다. <code>text</code>의 <code>type</code>이 <code>"word"</code>이고 <code>word</code>의 값은 각 글자의 annotation dictionary를 원소로 하는 list <code>[{'charbox': [68, 17, 107, 89], 'value': '가'}, {'charbox': [113, 25, 171, 91], 'value': '게'}]</code>이다.<br>

```json
{"info": {"name": "Korean OCR Data Set",
  "description": "Korean OCR Data Set (word handwrite)",
  "date_created": "2020-12-22 15:03:00"},
 "image": {"file_name": "00140001001.jpg",
  "width": 232,
  "height": 109,
  "dpi": 300,
  "bit": 24},
 "text": {"type": "word",
  "output": "handwrite",
  "word": [{"charbox": [68, 17, 107, 89], "value": "가"},
   {"charbox": [113, 25, 171, 91], "value": "게"}]},
 "license": {"output": "handwrite",
  "font": "",
  "font_no": "",
  "font_license": "",
  "font_url": "",
  "writer_no": "001",
  "writer_gender": "female",
  "writer_age": "40"}}
```  

이러한 형태의 JSON에서 원하는 GT 문자열을 추출해내서 lmdb로 변환하는 코드는 <code>create_lmdb_dataset_AI_Hub.py</code>이다.<br>

----------  

  
## Annotation as a JSON File 2  
<a href=https://aihub.or.kr/aidata/133> KoreanSTR 데이터셋 </a> 역시 JSON 형태로 annotation이 되어 있는데 JSON의 mapping 형태가 조금 다르다.<br>
annotation file인 handwriting_data_info1.json 파일의 형식은 다음과 같고

```bash
{'info':    , 
 'images': [...{'id': '00000003',
             'width': 3739,
             'height': 175,
             'file_name': '00000003.png',
             'license': 'AI 오픈 이노베이션 허브'},
            ...], 
 'annotations': [...{'id': '00000003',
                  'image_id': '00000003',
                  'text': '정권자였는데 이에 대한 사과가 우선"이라고 했다. 한편 문 후보와 김',
                  'attributes': {'type': '문장', 'gender': '여', 'age': '28', 'job': '직장인'}},
              ...], 
 'licenses': }
```

`images`에 이미지 파일에 관한 정보, `annotations`에 annotation에 관한 정보가 각각 dict들의 list 형태로 맵핑되어 있음을 알 수 있다. 그리고 `annotations`의 `image_id`가 `images`의 `id`에 해당하는 일종의 외래키 역할을 하기 때문에, `image_id`를 매개로 annotation 정보와 이미지 파일을 pairing할 수 있다.

lmdb로 변환하기 위한 도커 컨테이너 환경

```bash
docker pull seonwhee0docker/abinet:torch1.12
docker run -it --name abinet --ipc=host -v /Data1:/Data1 -v /Data2:/Data2 -v $(pwd)/ABINet:/app -p 8881:8888 seonwhee0docker/abinet:torch1.12 /bin/bash
```

컨테이너에서 jupyter notebook을 실행시키고 8881 포트로 접속한다.  

```bash
jupyter notebook --ip 0.0.0.0 --allow-root --no-browser
```
  
create_lmdb_dataset_KoreanSTR.py  
  
가장 시간이 많이 소요되는 것은 모든 파일을 어노테이션과 비교해가면서 pair를 만드는 `get_all_pairs` 메소드 중 `self.imgName2path_annot`를 실행하는 부분이기 때문에 `total_imgs_file`를 slicing해서 만개(`total_imgs_file[:10000]`)만 문제 없이 실행되는지 확인하고, 문제없으면 전체 이미지 `total_imgs_file`에 대해서 pair를 만들어준다.  

```python
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
```
  
lmdb_dataset 실행 명령  

```bash
python create_lmdb_dataset_KoreanSTR.py --outputPath='/Data2/Dataset/Preproc/ABINet/'
```
