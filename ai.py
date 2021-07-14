import tensorflow as tf
import numpy as np
import os

# 데이터셋 로딩
food = np.load('./result.npy')
np.random.shuffle(food)

row = food.shape[0]
train_num = int(row*0.7)

x_train = food[:train_num, :12288]
x_test = food[train_num:, :12288]
y_train = food[:train_num, 12288:]
y_test = food[train_num:, 12288:]

#print(x_train.shape)
# print(len(x_train))
x_train = x_train.reshape(-1, 64, 64, 3) # 첫번째 인자 = -1 이였음 4822

batch_size = 120 # 원래 500
training_epochs = 5

def  create_model():
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(64, 64, 1)),
        # tf.keras.layers.MaxPooling2D(),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        # tf.keras.layers.MaxPooling2D(),
        # tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        # tf.keras.layers.MaxPooling2D(),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', input_shape=(64, 64, 3)),#
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(15, activation='softmax') #relu
        # tf.keras.layers.Dense(1, activation='softmax')
    ])
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
# => flatten 사용
