import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OneHotEncoder

TRAIN_NUM = 35000
BATCH_SIZE = 32

trainData = pd.read_csv("../data/kaggle/digitRecognizer/train.csv")
testData = pd.read_csv("../data/kaggle/digitRecognizer/test.csv")

test_image = np.array(testData)
test_image = test_image / 255.0
test_num = len(test_image)

print(test_image.shape)

# 转换为onehot
labels = np.array(trainData.pop('label'))
labels = OneHotEncoder().fit_transform(labels.reshape(-1,1))
labels = labels.toarray()

images = np.array(trainData)
images = images / 255.0

x_train,x_valid = images[:TRAIN_NUM],images[TRAIN_NUM:]
y_train,y_valid = labels[:TRAIN_NUM],labels[TRAIN_NUM:]


x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])

x_image = tf.reshape(x,[-1,28,28,1])

w_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b_conv1 = tf.Variable(tf.zeros([32]))
h_conv1 = tf.nn.conv2d(x_image, w_conv1, strides=[1,1,1,1],padding="SAME")
h_conv1 = tf.nn.relu(h_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

w_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
b_conv2 = tf.Variable(tf.zeros([64]))
h_conv2 = tf.nn.conv2d(h_pool1,w_conv2,strides=[1,1,1,1],padding='SAME')
h_conv2 = tf.nn.relu(h_conv2)
h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

w_fc1 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
b_fc1 = tf.Variable(tf.zeros([1024]))
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

w_fc2 = tf.Variable(tf.truncated_normal([1024,10]))
b_fc2 = tf.Variable(tf.zeros([10]))
y = tf.matmul(h_fc1_drop, w_fc2)+ b_fc2

test_result = tf.argmax(y,1)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_accuracy = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_accuracy,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    validate_feed = {x: x_valid, y_: y_valid, keep_prob: 1.0}
    #test_feed = {x: test_image , y_: , keep_prob: 1.0}

    index = 0
    for i in range(10000):
        x_batch = x_train[index*BATCH_SIZE: (index+1)*BATCH_SIZE]
        y_batch = y_train[index*BATCH_SIZE: (index+1)*BATCH_SIZE]
        if (index+2)*BATCH_SIZE >= TRAIN_NUM:
            index=-1
        index += 1

        sess.run(train_step,feed_dict={x:x_batch,y_:y_batch,keep_prob:0.5})

        if i%1000 ==0:
            valid_accu = sess.run(accuracy,feed_dict=validate_feed)
            print("epoch %d ,acc: %f" %(i,valid_accu))


    result = np.zeros(testData.shape[0])
    for i in range(testData.shape[0]//100):
        result[i*100:(i+1)*100] = sess.run(test_result,feed_dict={x:test_image[i*100:(i+1)*100],keep_prob:1.0})

      #  print(result)

    result = result.astype(np.int32)
    saveDataFrame = pd.DataFrame({'ImageId':range(1,testData.shape[0]+1),'Label':result},columns=['ImageId', 'Label'])
    saveDataFrame.to_csv('0629.csv',index=False)
