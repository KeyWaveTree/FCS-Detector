#사람이 같이 포함된 이미지 불러오기
# 리펙토링 단계
# 데이터 다운로드(데이터 불러오기)
# 사람 랜드 마크 이용한 얼굴 블러처리 후 저장

import mediapipe as mp
import pandas as pd
import numpy as np
import cv2 as cv
import os

from PIL import Image, ImageDraw

data_path = os.path.join('../Clothing_Data')
# #디렉토리도 루트를 지정해야 한다.
# #루트가 있어야 아레의 내용을 확인 할 수 있는데 지금 경로만 하여 밑에 있는 애들을 못찾은 것이다.

file_dict = {}
#파일 순차적으로 리스트 뽑기
def img_path_list1(origin_path : str)->dict:
      for (root, directories, files) in os.walk(origin_path):
            print('디렉토리')
            for d in directories:
                  d_path = os.path.join(root, d)
                  print(d_path)

            print('파일들')
            for file in files:
                  file_path = os.path.join(d, file)
                  print(file, d)
                  if d not in file_dict:
                        file_dict[d]=[]
                  file_dict[d].append(file_path)


# #path에서
# #파일 순차적으로 리스트 뽑기
def img_path_list2(origin_path : str)->dict:
      for (root, directories, files) in os.walk(origin_path):
            for name in files:
                  print(os.path.join(root, name))
            print()

img_path_list1(data_path)
img_path_list2(data_path)

#파일 순서 목록을
#먼저 사진 하나로 실험

#1 사진에서 랜드 마크 찍기 다른 폴더 사진으로 저장

img_pose = mp.solutions.pose #미디어 파이프 솔루션


test_path='../Clothing_Data/zalando/shirt/0DB22O00B-Q11@8.jpg'
#경로 세팅
if not os.path.exists('../ImgProcess'):
      os.mkdir('../ImgProcess')

if not os.path.exists('../ImgProcess/Clothing_Landmark'):
      os.mkdir('../ImgaeProcess/Clothing_Landmark')

with img_pose.Pose(
      static_image_mode=True,
      model_complexity=2,
      enable_segmentation=True,
      min_detection_confidence=0.5) as pose:

