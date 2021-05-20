import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
# matplotlib.matplotlib_fname()



#rc['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
import time
import random

#import font to use at the outcome window
#rc('font', familly='AppleGothic')


# food = np.load('food64.npy')
food = np.load('./result.npy')

np.random.shuffle(food)

row = food.shape[0]
train_num = int(row*0.7)
#  print(train_num) 7459개
# 총 12432
# print(food.shape[0])

# 64*64*3 = 12288
x_train = food[:train_num, :12288]
x_test = food[train_num:, :12288]
# print(x_train.shape)
y_train = food[:train_num, 12288:]
y_test = food[train_num:, 12288:]
# print(y_train.shape)


learning_rate = 0.001
batch_size = 10
training_epochs = 10

global_step = tf.Variable(0, trainable=False, name='global_step')
keep_prob = tf.placeholder(tf.float32)

#모델 설계
# X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
X = tf.placeholder(tf.float32, [None, 12288])
x_img = tf.reshape(X, [-1, 64, 64, 3])
Y = tf.placeholder(tf.float32, [None, 3])


W1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
b1 = tf.Variable(tf.constant(0.1, shape = [64]))
L1 = tf.nn.relu(tf.nn.conv2d(x_img, W1, strides = [1,1,1,1], padding='SAME')+b1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)


W2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2))
b2 = tf.Variable(tf.constant(0.1, shape = [64]))
L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides = [1,1,1,1], padding='SAME') + b2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)


W3 = tf.Variable(tf.random_normal(shape=[3, 3, 64,128], stddev=5e-2))
b3 = tf.Variable(tf.constant(0.1, shape = [128]))
L3 = tf.nn.relu(tf.nn.conv2d(L2, W3, strides = [1,1,1,1], padding='SAME')+b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)


W4 = tf.Variable(tf.random_normal(shape=[3,3, 128,128], stddev=5e-2))
b4 = tf.Variable(tf.constant(0.1, shape = [128]))
L4 = tf.nn.relu(tf.nn.conv2d(L3, W4, strides = [1,1,1,1], padding='SAME')+b4)
L4 = tf.nn.dropout(L3, keep_prob=keep_prob)


W5 = tf.Variable(tf.random_normal(shape=[3,3, 128,128], stddev=5e-2))
b5 = tf.Variable(tf.constant(0.1, shape = [128]))
L5 = tf.nn.relu(tf.nn.conv2d(L4, W5, strides = [1,1,1,1], padding='SAME')+b5)
L5 = tf.nn.dropout(L3, keep_prob=keep_prob)
L5 = tf.nn.max_pool(L5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#fully_connected1
L5_flat = tf.reshape(L5, [-1,128*8*8])


fc_W1 = tf.get_variable("W4", shape=[128*8*8, 384],
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
fc_b1 = tf.Variable(tf.random_normal([384]))
fc_L1 = tf.nn.relu(tf.matmul(L5_flat, fc_W1)+ fc_b1)
fc_L1 = tf.nn.dropout(fc_L1, keep_prob=keep_prob)


fc_W2 = tf.get_variable("W5", shape=[384, 3],
                     initializer=tf.truncated_normal_initializer(stddev=0.1))
fc_b2 = tf.Variable(tf.random_normal([3]))
logits = tf.matmul(fc_L1, fc_W2) + fc_b2
y_pred = tf.nn.softmax(logits)
# print(y_pred.get_shape())


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

count = int(train_num / batch_size)
if train_num % batch_size != 0:
    count += 1


with tf.Session() as sess:

    saver = tf.train.Saver(tf.global_variables())
    # checkpoint
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restore!")
    else:
        sess.run(tf.global_variables_initializer())
        print("make new data")


    print('Learning started. It takes sometime.')

    learning_start = time.time()
    learning_start_time = time.strftime("%X", time.localtime())


    for i in range(training_epochs):

        for b in range(count):
            b_count = b*batch_size
            # batch = next_batch(b_count,b_count+batch_size, , y_train)
            x = x_train[b_count:b_count+batch_size,:]
            y = y_train[b_count:b_count+batch_size,:]
            # print("b    ",b)

            if b == count-1 :
                x = x_train[b_count:,:]
                y = y_train[b_count:,:]

            feed_dict = {X : x, Y : y, keep_prob: 0.5}
            a, c, _ = sess.run([accuracy, cost, optimizer], feed_dict=feed_dict)

        print('Epoch:', '%04d' % (i), 'cost =', '{:.9f}'.format(c), ' accuracy =', '{:.9f}'.format(a))
        # saver.save(sess, '~/ckpt_food/cnn_food.ckpt', global_step=global_step)

    # save the model
    save_path = saver.save(sess, './model/ckpt_food')
    print('Learning Finished!')

    learning_end = time.time()
    learning_end_time = time.strftime("%X", time.localtime())
    min = (learning_end - learning_start)/60
    print('%s ~ %s, 소요시간: %s분' %(learning_start_time, learning_end_time, '{:.1f}'.format(min)))

    # Test model and check accuracy

    # if you have a OOM error, please refer to lab-11-X-mnist_deep_cnn_low_memory.py

    #테스트 데이터셋을 집어 넣습니다.
    testing_epoch = 5


    x_show = 0
    y_show = 0
    for i in range(testing_epoch):

        for b in range(count):

            b_count = b * batch_size
            x = x_train[b_count:b_count+batch_size,:]
            y = y_train[b_count:b_count+batch_size,:]

            if b == count-1 :

                x_show = x_train[b_count:,:]
                y_show = y_train[b_count:,:]
                break

            acc = sess.run(accuracy, feed_dict={X: x, Y: y, keep_prob: 1.0})
            print('Accuracy:','{:.2f}'.format(acc*100), '%')


    #사진으로 보여주기
    #len => 리스트 사이즈 range=> 지정한 숫자만큼 배열을 만들어줌
    labels = sess.run(logits, feed_dict={X: x_show, keep_prob: 1})

    random_idxs = random.sample(range(len(food)), 5)

    fig = plt.figure()
    for i, r in enumerate(random_idxs):
        subplot = fig.add_subplot(2, 5, i+1)
        subplot.set_xticks([])
        subplot.set_yticks([])

        #분류할 음식 목록입니다. 순서대로 인덱스를 붙여줍니다.
        food_list = ['carrot','egg','onion']

        #food_list = ['Bread','Carrot','Cheese','Egg','Ham','Lettuce','Onion','Patty','Pickle','Potato'
           # ,'Radish','Rib','Shiitake_mushroom','Tomato','Welsh_onion']

        #subplot.set_title(food_list[np.argmax(labels[i])])
        subplot.set_title(food_list[np.argmax(y_test[i])])
        # print(food_list[np.argmax(labels[i])])

        # if x_test[i].dtype != np.uint8:
        #     x_test[i] = (255*x_test[i]).astype(np.uint8)

        # dtype=float (and compatible dims)
        subplot.imshow(np.float64((x_test[i].reshape(64, 64,3))/255.0))

    plt.show()

  # Save the variables to disk.
