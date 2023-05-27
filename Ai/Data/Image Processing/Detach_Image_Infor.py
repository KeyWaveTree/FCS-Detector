#완성파일 
 
#사람이 같이 포함된 이미지 불러오기
# 리펙토링 단계
# 데이터 다운로드(데이터 불러오기)
# 사람 랜드 마크 이용한 얼굴 블러처리 후 저장

import numpy as np
import pandas as pd
import cv2 as cv
import os

# #디렉토리도 루트를 지정해야 한다.
# #루트가 있어야 아레의 내용을 확인 할 수 있는데 지금 경로만 하여 밑에 있는 애들을 못찾은 것이다.
data_path = os.path.join('../Clothing_Data')

#폴더 안 이미지 파일 순차적으로 딕셔너리 얻기
def img_path_list(origin_path : str)->dict:
      file_dict = {}
      # 디렉토리를 재귀로 탐색에서 파일만 추출하는 반복문
      for (root, directories, files) in os.walk(origin_path): #각 루트, 디렉토리, 파일들
            #print('파일들')
            for file in files:
                  file_path = os.path.join(root, file)
                  #print(file, root)
                  if root not in file_dict:
                        file_dict[root]=[]
                  file_dict[root].append(file_path)

      return file_dict

print(img_path_list(data_path))
#이미지 처리
#1. 사람 랜드마크 찍어내기
#2. 사람 사진 태두리 분리


#이미지 처리 후 data 파일 저장

