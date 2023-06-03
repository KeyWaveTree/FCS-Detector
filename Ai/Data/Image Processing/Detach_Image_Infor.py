#완성파일 
 
#사람이 같이 포함된 이미지 불러오기
# 리펙토링 단계
# 데이터 다운로드(데이터 불러오기)
# 사람 랜드 마크 이용한 얼굴 블러처리 후 저장

import mediapipe as mp
import pandas as pd
import numpy as np
import cv2
import os

# #디렉토리도 루트를 지정해야 한다.
# #루트가 있어야 아레의 내용을 확인 할 수 있는데 지금 경로만 하여 밑에 있는 애들을 못찾은 것이다.
Image_Data_Path = os.path.join('../Clothing_Data')

class FCS_ImageProcess:
      def __init__(self, data_path : str):
            self.data_path = data_path

#폴더 안 이미지 파일 순차적으로 딕셔너리 얻기
def img_path_map(origin_path : str)->dict:
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

mp_img_pose = mp.solutions.pose #미디어 파이프 솔루션
mp_pose_img_drawing=mp.solutions.drawing_utils

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
img_path = '../ImgProcess/Clothing_Landmark/' + key_path + '/' + key_name

#pass
def img_contour(get_img_path: str)->cv2: #path

      #근데 이미지 확장성이 jpg라서 용량이 큼 용량을 줄일 필요가 있지 않을까?
      #if of openCV
      #이미지 처리를 할때 우선적으로 관심영역만 표시를 할까?
      #아니면 원본에서 처리를 한다음에 인식을 해야 할까?
      origin_img=cv2.imread(get_img_path,cv2.COLOR_BGR2HSV) #이미지 읽어오기
      #자른다음 하자 #노이즈제거는 GRAY이지만


      gaussian_blur =cv2.GaussianBlur(origin_img ,ksize=(3,3),sigmaX=0)

      thresh_img= np.median(gaussian_blur)
      thresh_lower=int(max(0,(1.0 - 0.55) *thresh_img))
      thresh_upper=int(max(255, (1.0 + 0.55)*thresh_img))

      #케니필터로
      cv2.adaptiveThreshold(gaussian_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY_INV, 9 ,10)
      edge=cv2.Canny(gaussian_blur, threshold1=thresh_lower, threshold2=thresh_upper)

      #binary_img=cv2.bitwise_or(nagative_binary_img, positive_binary_img, edge)
      #
      # _, nagative_img_binary = cv2.threshold(nagative_img, 150, 255, cv2.THRESH_BINARY)
      # _, positive_img_binary = cv2.threshold(positive_img, 150, 255, cv2.THRESH_BINARY)
      #
      # nagative_find, nagative_hier = cv2.findContours(nagative_img_binary,
      #                                                 mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
      # positive_find, positive_hier = cv2.findContours(positive_img_binary,
      #                                                 mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
      #
      # index=0
      # while index>=0:
      #       #사람
      #       #nagavite
      #
      #       nagative_img_contour=cv2.drawContours(origin_img,
      #                                    contours=nagative_find, contourIdx=index,
      #                                    color=(0, 0, 255), lineType=cv2.LINE_8, hierarchy=nagative_hier)
      #       #positive
      #
      #
      #       positive_img_contour = cv2.drawContours(origin_img,
      #                                      contours=positive_find, contourIdx=index,
      #                                      color=(0, 255, 0), lineType=cv2.LINE_8, hierarchy=positive_hier)
      #
      #       img_contour=cv2.bitwise_and(nagative_img_contour, positive_img_contour)

      return edge

def people_img_landmark(get_origin_img:cv2)->cv2:
      with mp_img_pose.Pose(
      static_image_mode=True,
      enable_segmentation=True,
      min_detection_confidence=0.5) as pose:

            test_img = cv2.cvtColor(get_origin_img, cv2.COLOR_BGR2RGB)

            img_pose = pose.process(test_img)

            landmarks = img_pose.pose_landmarks.landmark #각 랜드마크의 활성화 상태 리스트
            left_shoulder =[landmarks[mp_img_pose.PoseLandmark.LEFT_SHOULDER.value].x, #왼쪽어깨 랜드마크 x좌표
                            landmarks[mp_img_pose.PoseLandmark.LEFT_SHOULDER.value].y] #왼쪽어깨 랜드마크 y좌표

            right_shoulder = [landmarks[mp_img_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_img_pose.PoseLandmark.RIGHT_SHOULDER.value].y ]
            print(left_shoulder, right_shoulder)

            reset_test_img = cv2.cvtColor(test_img, cv2.cv2.COLOR_RGB2BGR)

            mp_pose_img_drawing.draw_landmarks(reset_test_img, img_pose.pose_landmarks, mp_img_pose.POSE_CONNECTIONS,
                                    mp_pose_img_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,circle_radius=2),
                                    mp_pose_img_drawing.DrawingSpec(color=(245, 66, 230),thickness=2, circle_radius=2))

      return reset_test_img
#지금 모든 경로는 안씀
path=img_path_map(Image_Data_Path)

origin_image= cv2.imread(test_path)

people_landmark = people_img_landmark(origin_image)
cv2.imshow(img_path, people_landmark)
cv2.imwrite(img_path + '.jpg', people_landmark)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 사람 태두리 and 옷 패턴 윤곽처리
# 찍은 사진을 분석
