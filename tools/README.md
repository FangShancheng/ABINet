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
문제는 annotation file이 위와 같은 형식인데, label 부분이 문자열이 아니고 숫자로 되어 있음을 알 수 있다. 오히려 정답 문자열은 파일이름에 들어 있다. 따라서 이 경우는 파일 이름에서 label을 추출하는 코드를 추가한 `create_lmdb_dataset_MJ.py` 를 사용해서 데이터셋을 만들어 주어야 한다.
<br>
```shell  
python tools/create_lmdb_dataset_MJ.py --inputPath=/home/ubuntu/Dataset/text_recognition/Korean/mjsynth --outputPath=/home/ubuntu/Playground/ABINet/data/training/MJ/MJ_test --gtFile=/home/ubuntu/Dataset/text_recognition/Korean/mjsynth/annotation_test.txt  
```
<br>  
반면 별도의 annotation 파일은 없고, 이미지 파일이름을 라벨로 하는 데이터셋도 있다.<br>   

```
/home/ubuntu/Dataset/custom/라면.jpg  --> 라면
```
<br>
이 경우는 `create_lmdb_dataset_var.py`을 실행시키면 된다.

