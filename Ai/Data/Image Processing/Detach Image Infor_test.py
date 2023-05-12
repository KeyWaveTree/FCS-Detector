#사람이 같이 포함된 이미지 불러오기
# 리펙토링 단계
# 데이터 다운로드(데이터 불러오기)
# 사람 랜드 마크 이용한 얼굴 블러처리 후 저장

import numpy as np
import pandas as pd
import cv2 as cv
import os

#파일 순차적으로 리스트 뽑기
def img_path_list(origin_path : str)->list:
      all_cloth_folder = os.listdir(origin_path)
      return all_cloth_folder


print(img_path_list('../Clothing_Data/zalando'))

#파일 순서 목록을