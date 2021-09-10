import io

from flask import Flask, render_template, request, url_for
from flask import jsonify
# from connection import get_connection, get_s3_connection
#import base64
#import cv2
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import requests
from PIL import Image
from io import BytesIO
import time

init_Base64 = 21
label_dict = {0:'아보카도', 1:'바나나', 2:'블루베리', 3:'가지', 4:'생강', 5:'대파', 6:'키위', 7:'레몬', 8:'오렌지', 9:'복숭아', 10:'고구마', 11:'감자', 12:'토마토', 13:'계란', 14:'고등어', 15:'고르곤졸라치즈', 16:'김치', 17:'까망베르치즈', 18:'꼬막', 19:'느타리버섯', 20:'단감', 21:'단호박', 22:'닭고기', 23:'당근', 24:'딸기', 25:'떡', 26:'만두', 27:'명란젓', 28:'미역', 29:'밤', 30:'베이글',
              31:'사과', 32:'새우', 33:'오징어', 34:'청양고추'}

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 3

app = Flask(__name__)

# @app.route('/camera')
# def index():
#     return render_template('index.html')

@app.route('/camera/predict', methods=['post']) # 'get'
def make_prediction():
    # s3_connection = get_s3_connection()
    # image_file = request.files['image1'] # file로 보내기

    # img url로 받기
    # url = "https://img.hankyung.com/photo/202012/99.24812305.1.jpg"
    url = request.args['img_url']
    start = time.time()
    res = requests.get(url)
    # print(url)
    # print(res)
    print(time.time() - start)

    image_file = BytesIO(res.content)

    if not image_file:
        #return render_template('index.html', label="no files")
        return jsonify({
            "label": 'error',
        })

    pil_img = Image.open(image_file)

    pil_img = pil_img.resize((IMAGE_HEIGHT, IMAGE_WIDTH), Image.BILINEAR)
    pil_img = np.array(pil_img)

    if len(pil_img.shape) == 2:  # Black and white
        pil_img = pil_img.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        pil_img = np.repeat(pil_img, 3, axis=3)

    pil_img = pil_img.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    pil_img = tf.cast(pil_img, tf.float32) ## 추가

    y_predict = model.predict(pil_img)
    # print(y_predict)

    label = label_dict[y_predict[0].argmax()]
    index = str(np.squeeze(y_predict)) # 위치확인
    print(index)
    confidence = y_predict[0][y_predict[0].argmax()]
    lb = '{} {:.2f}%'.format(label, confidence * 100)

    # return render_template('index.html', label=lb)
    return jsonify({
        "label": label,
        "probability": lb
    })


if __name__ == '__main__':
    model = load_model("food_classifer.h5")
    if model:
        print('model load success')
    app.run(port=4500, debug=True) # host='0.0.0.0' => 외부에서 접근 가능

#### model load 못해서 predict가 안됨. => h5파일 저장해서 해결

