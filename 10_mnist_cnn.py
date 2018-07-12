import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/mnist",one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])

x_image = tf.reshape(x,[-1,28,28,1])

W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b_conv1 = tf.Variable(tf.zeros([32]))
h_conv1 = tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding='SAME')
h_conv1 = tf.nn.relu(h_conv1)
h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

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

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_accuracy = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_accuracy,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels,keep_prob:1.0}
    test_feed = {x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}

    for i in range(10000):
        x_batch, y_batch = mnist.train.next_batch(32)
        sess.run(train_step,feed_dict={x:x_batch,y_:y_batch,keep_prob:0.5})

        if i % 1000 ==0:
            valid_accuracy = sess.run(accuracy,feed_dict=validate_feed)
            print("epcho %d, acc: %f" %(i,valid_accuracy))

    test_accu = sess.run(accuracy,feed_dict=test_feed)
    print("test accu: %f" % test_accu)




