from keras.layers import Conv2D, MaxPool2D, Flatten

import tensorflow as tf
import numpy as np
import os

# 데이터셋 로딩
food = np.load('../result.npy')
np.random.shuffle(food)

row = food.shape[0]
train_num = int(row*0.7)

x_train = food[:train_num, :12288]
x_test = food[train_num:, :12288]

y_train = food[:train_num, 12288:]
y_test = food[train_num:, 12288:]

# print(x_train.shape)
# print(len(x_train))

# reshape(총 샘플 수, 1차원 속성의 수)
x_train = x_train.reshape(-1, 64, 64, 3) # 첫번째 인자 = -1 이였음 4822

batch_size = 120 # 원래 500
training_epochs = 50

# sequential 딥러닝구조,층설정 / compile 정해진모델 컴파일 / fit 모델 실제 수행
def  create_model():
    model = tf.keras.models.Sequential([

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='sigmoid'),

        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01)),
        tf.keras.layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01)),

        tf.keras.layers.Dense(64, kernel_initializer='orthogonal'),  # 커널을 랜덤한 직교행렬로 초기화
        tf.keras.layers.Dense(512, activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.Dense(64, activation='relu'),

        tf.keras.layers.Conv2D(64, 3, 3, input_shape=(64, 64, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),

        #tf.keras.layers.Conv2D(64, 3, 3, input_shape=(64, 64, 3), activation='relu'),
        #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #tf.keras.layers.Flatten(),

        # Hidden layer
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(11, activation='softmax')  # 클래수 개수
    ])

    # 현재 출력 형상을 포함하여 지금까지의 모델 요약표시
    #model.summary()

    # 모델 실행 환경 설정
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', # sparse_
                  metrics=['accuracy'])

    return model

model = create_model()
# model.summary()

# (훈련데이터, 레이블데이터, 에포크, 배치크기)
model.fit(x_train, y_train, epochs=training_epochs, batch_size=batch_size, verbose=1)

# model.evaluate(x_test, y_test, batch_size=batch_size)
model.save('food_classifer.h5')

# 모델 최적화 설정

# # 가중치 불러오기
# checkpoint_path = "./model"
# checkpoint_dir = os.path.dirname(checkpoint_path)# Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                    save_weights_only=True,
#                                    verbose=1)# Train the model with the new callback
# model.fit(x_train,
#           y_train,
#           epochs=10,
#           validation_data=(x_test,y_test),
#           callbacks=[cp_callback])
# # 오류: tensorflow.python.framework.errors_impl.InvalidArgumentError:  Incompatible shapes: [32,64,64,1] vs. [32,15]
# # 	 [[node Equal (defined at /Users/ji_ji/OneDrive/바탕 화면/냉부해/4our_AI/ai.py:61) ]] [Op:__inference_train_function_960]