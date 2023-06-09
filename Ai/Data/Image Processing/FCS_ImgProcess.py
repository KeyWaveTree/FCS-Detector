# 완성파일

# 사람이 같이 포함된 이미지 불러오기
# 리펙토링 단계
# 데이터 다운로드(데이터 불러오기)
# 사람 랜드 마크 이용한 얼굴 블러처리 후 저장

import mediapipe as mp
import numpy as np
import cv2
import os

# #디렉토리도 루트를 지정해야 한다.
# #루트가 있어야 아레의 내용을 확인 할 수 있는데 지금 경로만 하여 밑에 있는 애들을 못찾은 것이다.
mp_img_pose = mp.solutions.pose

#초기화 세팅
def init_path(edge_path, img_path):
    if not os.path.exists('../ImgProcess'): os.mkdir('../ImgProcess')
    if not os.path.exists(edge_path): os.mkdir(edge_path)
    if not os.path.exists(img_path): os.mkdir(img_path)

# 폴더 안 이미지 파일 순차적으로 딕셔너리 얻기
def image_path_map(origin_path: str) -> dict:
    file_dict = {}
    # 디렉토리를 재귀로 탐색에서 파일만 추출하는 반복문
    for (root, directories, files) in os.walk(origin_path):  # 각 루트, 디렉토리, 파일들
        # print('파일들')
        for file in files:
            file_path = os.path.join(root, file)
            # print(file, root)
            if root not in file_dict:
                file_dict[root] = []
            file_dict[root].append(file_path)

    return file_dict

def canny_edge_img(origin_img):
    roi = origin_img.copy()
    glay_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(glay_img, ksize=(3, 3), sigmaX=0)

    thresh_img = np.median(gaussian_blur)
    thresh_lower = int(max(0, (1.0 - 0.22) * thresh_img))
    thresh_upper = int(max(255, (1.0 + 0.22) * thresh_img))

    # 케니필터로
    cv2.adaptiveThreshold(gaussian_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 10)
    edge = cv2.Canny(gaussian_blur, threshold1=thresh_lower, threshold2=thresh_upper)

    return edge

def landmark_img_position(origin_img) -> dict or int:
    copy_img = origin_img.copy()
    with mp_img_pose.Pose(
    static_image_mode=True,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
        try:
            color_copy_img = cv2.cvtColor(copy_img, cv2.COLOR_RGB2BGR)
            img_pose = pose.process(color_copy_img)
            landmarks = img_pose.pose_landmarks.landmark  # 각 랜드마크의 활성화 상태 리스트

            origin_img_height, origin_img_width, _ = color_copy_img.shape
            img_shape = [origin_img_width, origin_img_height]

            left_shoulder = (landmarks[mp_img_pose.PoseLandmark.LEFT_SHOULDER].x,
                             landmarks[mp_img_pose.PoseLandmark.LEFT_SHOULDER].y)
            right_shoulder = (landmarks[mp_img_pose.PoseLandmark.RIGHT_SHOULDER].x,
                              landmarks[mp_img_pose.PoseLandmark.RIGHT_SHOULDER].y)

            left_mouth = (landmarks[mp_img_pose.PoseLandmark.MOUTH_LEFT.value].x,
                          landmarks[mp_img_pose.PoseLandmark.MOUTH_LEFT.value].y)
            right_mouth = (landmarks[mp_img_pose.PoseLandmark.MOUTH_RIGHT.value].x,
                           landmarks[mp_img_pose.PoseLandmark.MOUTH_RIGHT.value].y)

            left_pinky = (landmarks[mp_img_pose.PoseLandmark.LEFT_PINKY.value].x,
                          landmarks[mp_img_pose.PoseLandmark.LEFT_PINKY.value].y)

            shoulder = {'l': np.multiply(left_shoulder, img_shape).astype(int),
                        'r': np.multiply(right_shoulder, img_shape).astype(int)}
            mouth = {'l': np.multiply(left_mouth, img_shape).astype(int),
                     'r': np.multiply(right_mouth, img_shape).astype(int)}

            right_elbow_x = np.multiply(landmarks[mp_img_pose.PoseLandmark.RIGHT_ELBOW].x, img_shape[0]).astype(int)
            left_pinky = np.multiply(left_pinky, img_shape).astype(int)

            middle_neck_y = (((shoulder['l'][1] + shoulder['r'][1]) // 2) + ((mouth['l'][1] + mouth['r'][1]) // 2)) // 2

            return {'x1': right_elbow_x - 5, 'y1': middle_neck_y, 'x2': left_pinky[0], 'y2': left_pinky[1]}

        except AttributeError:
            return -1

        except cv2.error:
            return -1

class FCS_ImageProcess:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.Image_Data_Path = os.path.join('../Clothing_Data')
        self.img_path_map = image_path_map(self.Image_Data_Path)

    def image_process(self):
        for img_path_key in self.img_path_map:
            key_name = 0; edge_path = '../ImgProcess/Clothing_Egde/'; img_split_path = '../ImgProcess/Clothing_Wear/'
            init_path(edge_path=edge_path, img_path=img_split_path)

            for img_path in self.img_path_map[img_path_key]:

                key_path = img_path.split('\\')[2]
                edge_write_path = edge_path + key_path + '/%s_%d.jpg' % (key_path, key_name)
                img_write_path = img_split_path + key_path + '/%s_%d.jpg' % (key_path, key_name)

                origin_img = cv2.imread(img_path)
                # 경로 세팅
                if not os.path.exists(edge_path + key_path): os.mkdir(edge_path + key_path)
                if not os.path.exists(img_split_path + key_path): os.mkdir(img_split_path + key_path)

                if landmark_img_position(img_path) == -1 or os.path.exists(edge_write_path) or os.path.exists(
                        img_write_path):
                    key_name += 1
                    continue

                else:
                    landmark_img = landmark_img_position(origin_img)
                    edge_img = canny_edge_img(origin_img)

                    setting_origin_img = origin_img[landmark_img['y1']:landmark_img['y2'],
                                                    landmark_img['x1']:landmark_img['x2']]
                    setting_edge_img = edge_img[landmark_img['y1']:landmark_img['y2'],
                                                landmark_img['x1']:landmark_img['x2']]
                    try:
                        cv2.imwrite(img_write_path, setting_origin_img)
                        cv2.imwrite(edge_write_path, setting_edge_img)

                    except cv2.error:
                        key_name += 1
                        continue

                print(f'{key_path} 이미지 다운로드 중(' + str(key_name + 1) + '/' +
                      str(len(self.img_path_map[img_path_key])) + '): ' + str(key_name))

                # 처리한 이미지 저장
                key_name += 1

        print('이미지 처리 완료')
        cv2.waitKey(0)
        cv2.destroyAllWindows()