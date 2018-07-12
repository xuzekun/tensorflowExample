import tensorflow as tf
import utils
import matplotlib.pyplot as plt
import time

DATA_FILE = 'birth_life_2010.txt'

# 1
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# 2
#X, Y = None, None
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

iterator = dataset.make_initializable_iterator()
X, Y = iterator.get_next()
#3
#w, b = None, None
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

#4
Y_predicted = w * X + b

#5
loss = tf.square(Y - Y_predicted, name='loss')

#6
LEARNING_RATE = 0.001
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)


start = time.time()

with tf.Session() as sess:
    #7
    writer = tf.summary.FileWriter('./graphs/linear_reg2', sess.graph)
    sess.run(tf.global_variables_initializer())

    #8
    for i in range(100):
        sess.run(iterator.initializer)
        total_loss = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
        except tf.errors.OutOfRangeError:
            pass
        print('epoch %i: %s' %(i, total_loss/n_samples))

    #9
    w_out, b_out = sess.run([w, b])
    writer.close()
print('Done: %f seconds' % (time.time() - start))


plt.plot(data[:,0], data[:,1],'bo', label='real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='predicted')
plt.legend()
plt.show()