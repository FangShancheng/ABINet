## 데이터 전처리  
  
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
반면 별도의 annotation 파일은 없고, 이미지 파일이름을 라벨로 하는 데이터셋도 있다.<br>   

```
/home/ubuntu/Dataset/custom/라면.jpg  --> 라면
```
<br>
이 경우는 <code>create_lmdb_dataset_var.py</code>을 실행시키면 된다.
<br>
<br>
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
  "font": '',
  "font_no": '',
  "font_license": '',
  "font_url": '',
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
  "font": '',
  "font_no": '',
  "font_license": '',
  "font_url": '',
  "writer_no": "001",
  "writer_gender": "female",
  "writer_age": "40"}}
```  

이러한 형태의 JSON에서 원하는 GT 문자열을 추출해내서 lmdb로 변환하는 코드는 <code>create_lmdb_dataset_AI_Hub.py</code>이다.
