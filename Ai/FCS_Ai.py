#비슷한 이미지 보여주기 - 이미지 패턴의 가장 최고치 본인 제외
#케니는 먼저 이미지 분류
#원본은 이미지 연관
# 1. 이미지 태그 분류
# 2. 분류한 이미지에서 유사도 학습
import tensorflow as tf
import numpy as np
import glob,os

resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224), #랜덤크롭
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
])

#학습 데이터 로드 -데이터 수집, 가공은 이미 했기때문에 로드만 해준다.
def load_data():
    cloth_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'Data/ImgProcess',
        validation_split=0.2, #이미지의 80%를 훈련에 사용하고 20%를 유효성 검사
        subset='training',
        seed=123,
        image_size=(224, 224),
        batch_size=16
    )#모의고사 - 학습 데이터

    FCS_valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'Data/Clothing_Data',
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(224, 224),
        batch_size=16
    )#평가 문제

    rc_train_dataset = cloth_train_dataset.map(lambda x, y: (resize_and_crop(x), y)) #images, labels
    rc_valid_dataset = FCS_valid_dataset.map(lambda x, y: (resize_and_crop(x), y))#images, labels

    return rc_train_dataset, rc_valid_dataset


# 모델 생성
# 저장된 모델이 있으면 가져오고 없으면 그때 생성하게 함
def create_model():
    if os.path.exists('Data/Models/FCS'):
        model = tf.keras.models.load_model('Data/Models/FCS') #내 모델 있다면 그대로 불러옴

        model.layers[0].trainable = False #학습이 가능하게 만들것인가
        model.layers[2].trainable = True  #없어도 상관 없는데 불러와서 더 학습할 때 (중간 상태 저장해놓고 이어하기)를 위해서 씀
    else:
        #전이 학습
        model = tf.keras.applications.MobileNet(
            input_shape=(224, 224, 3),
            include_top=False,
            #가중치
            weights='imagenet',
        )

        model.trainable = False

        model = tf.keras.Sequential([
            model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1)
        ])

        learning_rate = 0.001
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            metrics=['cosine_similarity'] #코사인 유사도로 비슷한 이미지 수치 반환
        )

        cloth_train_dataset, FCS_valid_dataset = load_data()
        train_model(model, 100, cloth_train_dataset, FCS_valid_dataset, True)
    return model


# 모델 학습
#(학습할)모델,몇번할지,학습을 위한 트레인 데이터셋, 시험을 위한 데이터 셋,저장할지말지 결정하는 모델로 5개의 인자 받아옴
def train_model(model, epochs, cloth_train_dataset, FCS_valid_dataset, save_model):
    history = model.fit(cloth_train_dataset, epochs=epochs, validation_data=FCS_valid_dataset)
    if save_model: model.save('Data/Models/FCS')
    return history

# 학습된 모델로 예측
def predict(model, image):
    rc_image = resize_and_crop(np.array([image]))#np: 여러 이미지 동시에 예측가능하게 설계돼서 하나만 넣어도 np배열로 만들어서 넣어줘야 함
    result = model.predict(rc_image) #최종 가공된 데이터를 rc_image에 넣어줌
    if np.argmax(result) > 0.7: return result
    else: return None

#이미지 검사
def check_corrupted_images(folder_path='/Data/Clothing_Data'):
    num_skipped=0
    for (root, directories, files) in os.walk(folder_path):  # 각 루트, 디렉토리, 파일들
        # print('파일들')
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img=open(file_path,'rb')
                is_jfif = tf.compat.as_bytes("JFIF") in img.peek(10)
                print(f"{file}: OK")
            except (IOError, SyntaxError) as e:
                print(f"{file_path}: Corrupted ({str(e)})")
                num_skipped += 1
                # Delete corrupted image
                os.remove(file_path)
            finally:
                img.close()

    return num_skipped
if __name__ == '__main__':
    cloth_train_dataset,  FCS_valid_dataset = load_data()
    model = create_model()
    print(check_corrupted_images(), 'del')
    train_model(model, 5, cloth_train_dataset,  FCS_valid_dataset, True) #학습 누적된 모델 생성