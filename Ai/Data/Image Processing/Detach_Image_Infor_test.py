#사람이 같이 포함된 이미지 불러오기
# 리펙토링 단계
# 데이터 다운로드(데이터 불러오기)
# 사람 랜드 마크 이용한 얼굴 블러처리 후 저장
import mediapipe as mp
import pandas as pd
import numpy as np
import cv2
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

key_path = test_path.split('/')[3] #특정 옷 태그 뽑기
key_name = '1'

# 경로 세팅
if not os.path.exists('../ImgProcess'):
      os.mkdir('../ImgProcess')

if not os.path.exists('../ImgProcess/Clothing_Landmark'):
      os.mkdir('../ImgProcess/Clothing_Landmark')

if not os.path.exists('../ImgProcess/Clothing_Landmark/' + key_path):
      os.mkdir('../ImgProcess/Clothing_Landmark/' + key_path)

# 처리한 이미지 저장
landmark_img_path = '../ImgProcess/Clothing_Landmark/' + key_path + '/' + key_name
# 이미지 색상 저장 csv file로 변환
#이미지 사람과 옷 태두리
with img_pose.Pose(
static_image_mode=True,
enable_segmentation=True,
min_detection_confidence=0.5) as pose:

      #근데 이미지 확장성이 jpg라서 용량이 큼 용량을 줄일 필요가 있지 않을까?
      #if of openCV
      #이미지 처리를 할때 우선적으로 관심영역만 표시를 할까?
      #아니면 원본에서 처리를 한다음에 인식을 해야 할까?
      origin_img=cv2.imread(test_path) #이미지 읽어오기
      nagative_color_img=cv2.bitwise_not(origin_img)
      nagative_img =cv2.cvtColor(nagative_color_img, cv2.COLOR_BGR2GRAY) # 그레이 색상으로 하는 이유는 노이즈 제거와 속도 때문에
      positive_img=cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)# 그레이 색상으로 하는 이유는 노이즈 제거와 속도 때문에

      _, nagative_img_binary = cv2.threshold(nagative_img, 150, 255, cv2.THRESH_BINARY)
      _, positive_img_binary = cv2.threshold(positive_img, 150, 255, cv2.THRESH_BINARY)

      nagative_find, nagative_hier = cv2.findContours(nagative_img_binary,
                                                      mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
      positive_find, positive_hier = cv2.findContours(positive_img_binary,
                                                      mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
      #케니 필터를 이용해서 윤곽선 추출 -보류
      index=0
      while index>=0:
            #사람
            #nagavite

            nagative_img_contour=cv2.drawContours(origin_img,
                                         contours=find, contourIdx=index,
                                         color=(0, 0, 255), lineType=cv2.LINE_8, hierarchy=hier)
            #positive


            positive_img_contour = cv2.drawContours(origin_img,
                                           contours=find, contourIdx=index,
                                           color=(0, 255, 0), lineType=cv2.LINE_8, hierarchy=hier)

            img_contour=cv2.bitwise_and(nagative_img_contour, positive_img_contour)
      # roi = origin_img[y:y + h, x:x + w]
      #
      # # 옷
      # thres1 = np.random.randint(0, 50)
      # thres2 = np.random.randint(thres1, 100)
      # edge=cv2.Canny(roi, threshold1=thres1, threshold2=thres2)
      # 이미지 처리 후 처리용 폴더 경로에(새로운) 처리한 이미지 저장해야 함

      cv2.imshow(landmark_img_path, img_contour)  # 절대 경로, 이미지 읽을 때 옵션
      cv2.imwrite(landmark_img_path+'.jpg', img_contour)

      cv2.waitKey(0)
      cv2.destroyAllWindows()