from tensorflow import keras
import FCS_ImgProcess as prs
import FCS_Ai as ai
import numpy as np
import requests
import cv2
import os

#1. 케니 데이터 먼저 태그 찾기 - mnist
#2. 테그에 맞는 csv 파일 태그
#3. 이미지를 가지고 와서 유사도 검사
#4. csv에 저장 후 sort(내림차순)
#5. 웹사이트 return


def get_img_processing(web_img_path:str):
    #넘파이로 변환하여 다시 이미지로
    get_image_nparray = np.asarray(bytearray(requests.get(web_img_path).content), dtype=np.uint8)
    select_image = cv2.imdecode(get_image_nparray, cv2.IMREAD_COLOR)
    mnist_class_name = ['T-shirt/top', 'Trouser', 'Pullover',
                        'Dress', 'Coat', 'Sandal', 'Shirt',
                        'Sneaker', 'Bag', 'Ankle boot']

    landmark_img = prs.landmark_img_position(select_image) # 랜드마크 점찍기
    edge_img = prs.canny_edge_img(select_image)# 케니

    setting_color_img = select_image[landmark_img['y1']:landmark_img['y2'],
                         landmark_img['x1']:landmark_img['x2']]
    setting_edge_img = edge_img[landmark_img['y1']:landmark_img['y2'],
                       landmark_img['x1']:landmark_img['x2']]
    try:
        probability_model=keras.models.load_model('Data/Model/fm_cnn_model.h5')
        prob_color = probability_model.predict(setting_color_img)
        prob_edge=probability_model.predict(setting_edge_img)
        #np.argmax(np.maximum(prob_edge, prob_color)[0])- 태그 뽑기
        prs.img_path_routing(mnist_class_name[np.argmax(np.maximum(prob_edge,prob_color)[0])],
                             color_img=setting_color_img ,edge_img=setting_edge_img, answer_img=select_image)
    except cv2.error: pass

    cloth_detector_model = ai.create_model()
    predict=ai.predict(cloth_detector_model, setting_color_img)

    return predict
