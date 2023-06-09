# import mediapipe as mp
# import numpy as np
# import cv2
# import os
#
# mp_img_pose = mp.solutions.pose #미디어 파이프 솔루션
# mp_pose_img_drawing=mp.solutions.drawing_utils
#
# test_path='../Clothing_Data/zalando/shirt/0DB22O00B-Q11@8.jpg'
#
# key_path = test_path.split('/')[3] #특정 옷 태그 뽑기
# key_name = '1'
#
# # 경로 세팅
# if not os.path.exists('../ImgProcess'):
#       os.mkdir('../ImgProcess')
#
# if not os.path.exists('../ImgProcess/Clothing_Landmark'):
#       os.mkdir('../ImgProcess/Clothing_Landmark')
#
# if not os.path.exists('../ImgProcess/Clothing_Landmark/' + key_path):
#       os.mkdir('../ImgProcess/Clothing_Landmark/' + key_path)
#
# # 처리한 이미지 저장
# img_path = '../ImgProcess/Clothing_Landmark/' + key_path + '/'
#
# origin_img = cv2.imread(test_path)
# copy_img=cv2.imread(test_path)
# with mp_img_pose.Pose(
# static_image_mode=True,
# enable_segmentation=True,
# min_detection_confidence=0.5) as pose:
#     test_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)
#
#     img_pose = pose.process(test_img)
#
#     landmarks = img_pose.pose_landmarks.landmark  #각 랜드마크의 활성화 상태 리스트
#
#     origin_img_height, origin_img_width, _ = origin_img.shape
#     img_shape=[origin_img_width, origin_img_height]
#
#     left_shoulder = (landmarks[mp_img_pose.PoseLandmark.LEFT_SHOULDER].x ,
#                      landmarks[mp_img_pose.PoseLandmark.LEFT_SHOULDER].y )
#     right_shoulder=(landmarks[mp_img_pose.PoseLandmark.RIGHT_SHOULDER].x,
#                     landmarks[mp_img_pose.PoseLandmark.RIGHT_SHOULDER].y )
#
#     left_mouth=(landmarks[mp_img_pose.PoseLandmark.MOUTH_LEFT.value].x,
#                 landmarks[mp_img_pose.PoseLandmark.MOUTH_LEFT.value].y)
#     right_mouth=(landmarks[mp_img_pose.PoseLandmark.MOUTH_RIGHT.value].x,
#                  landmarks[mp_img_pose.PoseLandmark.MOUTH_RIGHT.value].y)
#
#     left_pinky=(landmarks[mp_img_pose.PoseLandmark.LEFT_PINKY.value].x,
#                 landmarks[mp_img_pose.PoseLandmark.LEFT_PINKY.value].y)
#
#     shoulder = {'l':np.multiply(left_shoulder, img_shape).astype(int),'r':np.multiply(right_shoulder,img_shape).astype(int)}
#     mouth = {'l':np.multiply(left_mouth , img_shape).astype(int),'r': np.multiply(right_mouth, img_shape).astype(int)}
#     right_elbow_x = np.multiply(landmarks[mp_img_pose.PoseLandmark.RIGHT_ELBOW].x, img_shape[0]).astype(int)
#     left_pinky = np.multiply(left_pinky, img_shape).astype(int)
#
#     middle_neck_y= (((shoulder['l'][1]+shoulder['r'][1])//2)+((mouth['l'][1] + mouth['r'][1]) // 2))//2
#
#     reset_test_img = cv2.cvtColor(test_img, cv2.cv2.COLOR_BGR2RGB)
#
#     mp_pose_img_drawing.draw_landmarks(reset_test_img, img_pose.pose_landmarks, mp_img_pose.POSE_CONNECTIONS,
#                                        mp_pose_img_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
#                                                                        circle_radius=2),
#                                        mp_pose_img_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
#                                                                        circle_radius=2))
#     #cv2.circle(reset_test_img, (right_elbow_x, middle_neck_y), radius=5, color=(225, 130, 130), thickness=1) - fin
#
# roi = origin_img[middle_neck_y:left_pinky[1], right_elbow_x :left_pinky[0]]
# glay_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# gaussian_blur =cv2.GaussianBlur(glay_img ,ksize=(3,3),sigmaX=0)
#
# thresh_img= np.median(gaussian_blur)
# thresh_lower=int(max(0,(1.0 - 0.22) *thresh_img))
# thresh_upper=int(max(255, (1.0 + 0.22)*thresh_img))
#
# #케니필터로
# cv2.adaptiveThreshold(gaussian_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY_INV, 9 ,10)
# edge=cv2.Canny(gaussian_blur, threshold1=thresh_lower, threshold2=thresh_upper)
#
# cv2.imshow(img_path+'1', edge)
# cv2.imshow(img_path+'3', reset_test_img)
#
# cv2.imwrite(img_path+'1' + '.jpg', edge)
# cv2.imwrite(img_path+'3' + '.jpg', reset_test_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()