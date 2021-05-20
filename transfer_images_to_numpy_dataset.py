import numpy as np
from PIL import Image
import os


def one_hot(i):
    a = np.zeros(3, 'uint8')
    a[i] = 1
    return a

data_dir = './Images/'
nb_classes = 3

result_arr = np.empty((1469, 12291)) # (전체 이미지 갯수, 64x64x3 + 15(클래스 갯수)(인덱스 one hot))
#2차원 백터 생성 [[],[],[]]과 같은 식으로 12432*12303 의 1차원 벡터가 아님! -> 이곳에 변환된 이미지 저장


idx_start = 0

for cls, food_name in enumerate(os.listdir(data_dir)):
    image_dir = data_dir + food_name + '/'
    file_list = os.listdir(image_dir)

    for idx, f in enumerate(file_list):
        im = Image.open(image_dir + f)
        pix = np.array(im)
        arr = pix.reshape(1, 12288)
        result_arr[idx_start + idx] = np.append(arr, one_hot(cls))
    idx_start += len(file_list)

np.save('result.npy', result_arr)
