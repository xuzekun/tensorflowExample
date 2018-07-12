""" Solution for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 1
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = 'data/mnist'
#utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
print(train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

print(test[0][0])

print(test[1][2])

import matplotlib.pyplot as plt



plt.imshow(test[0][1].reshape(28,28))
plt.imshow(test[0][2].reshape(28,28))
plt.show()



w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

#end = tf.nn.softmax(tf.matmul([test[0][0]], w) + b)

logits = tf.matmul(img, w) + b

print(test[0].shape)
print(test[0][0].shape)


aaa = tf.nn.softmax(tf.matmul(test[0], w) + b)

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss')

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

preds = tf.nn.softmax(logits)


correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))

accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    for i in range(n_epochs):
        sess.run(train_init)
        total_loss = 0
        n_batches = 0

        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        print('average loss epoch %s : %s' %(i, total_loss/n_batches))

    print('total time: %s' %(time.time() - start_time))

    sess.run(test_init)
    total_correct_preds = 0
    try:
        while True:
            print(sess.run([correct_pred]))
            accuracy_batch = sess.run(accuracy)
            print(accuracy_batch)

            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('accuarcy: %s' %(total_correct_preds/n_test))

    print('end')
    a = sess.run(aaa)
    print(a[1])
    print(a[2])