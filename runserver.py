from flask import Flask, render_template, request, url_for
from flask import jsonify
# from connection import get_connection, get_s3_connection
#import base64
#import cv2
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

init_Base64 = 21
label_dict = {0:'Apple Red', 1:'Avocado', 2:'banana', 3:'blueberry', 4:'egg', 5:'eggplant', 6:'ginger', 7:'greenonion', 8:'kiwi', 9:'lemon', 10:'orange', 11:'peach', 12:'potatosweet', 13:'potatowhite', 14:'tomato'}

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 3

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['post']) # 'get'
def make_prediction():
    # s3_connection = get_s3_connection()
    image_file = request.files['image1']
    # image_file = request.form.get('image1')
    # image_file = request.args['image1'] #file_name
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
        # "probability": l
    })



if __name__ == '__main__':
    model = load_model("food_classifer.h5")
    if model:
        print('model load success')
    app.run(port=4500, debug=True) # host='0.0.0.0' => 외부에서 접근 가능

#### model load 못해서 predict가 안됨. => h5파일 저장해서 해결

