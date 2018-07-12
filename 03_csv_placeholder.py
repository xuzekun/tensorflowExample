import tensorflow as tf

import csv

rows = csv.reader(open('data/train.csv','r',encoding='big5'))

x = list()
y = list()

n_rows = 0
for row in rows:
    if n_rows % 18 == 9:
        for i in range(3,27):
            x.append(float(row[i]))
    elif n_rows % 18 == 10:
        for i in range(3,27):
            y.append(float(row[i]))
    n_rows += 1


x_test = list()
y_test = list()

rows = csv.reader(open('data/test.csv','r',encoding='big5'))

n_rows = 0
for row in rows:
    if n_rows % 18 == 8:
        x_test.append(float(row[10]))
    n_rows += 1

rows = csv.reader(open('data/ans.csv','r',encoding='big5'))

n_rows = 0
for row in rows:
    if n_rows != 0:
        y_test.append(float(row[1]))
    n_rows +=1
print(y_test)



X = tf.placeholder(tf.float32, name='x')
Y = tf.placeholder(tf.float32, name='y')
X_test = tf.placeholder(tf.float32, name='xtest')
Y_test = tf.placeholder(tf.float32,name='ytest')

w = tf.get_variable(name='weights', shape=(1,1))
b = tf.get_variable(name='bias', shape=(1,1))

Y_ = X * w +b
loss = tf.reduce_sum(tf.square(Y_ - Y))

YY = X_test * w +b
lll = tf.reduce_sum(tf.abs(YY - Y_test))
ll2 = lll/ tf.cast(tf.shape(X_test), tf.float32)

len1 = tf.shape(X_test)
len2 = tf.shape(YY)
len3 = tf.shape(Y_test)

learning_rate = 0.00000001
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#x_test = [1,2,3,4,5]
#y_test = [0,0,0,0,0]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _, l = sess.run([optimizer, loss],feed_dict={X:x, Y:y})
        print(l)

    w_out, b_out = sess.run([w,b])

    gg,gg2, yy = sess.run([lll,ll2,YY],feed_dict={X_test:x_test,Y_test:y_test})
    print(yy)
    print(gg)
    print(gg2)

    l1,l2, l3 = sess.run([len1,len2,len3],feed_dict={X_test:x_test, Y_test:y_test})
    print(l1)
    print(l2)
    print(l3)

import matplotlib.pyplot as plt


c = x*w_out+b_out

plt.plot(x,y,'.')
plt.plot(x,c[0])

#plt.show()


import math
import numpy

yy1 = yy-y_test
uu = numpy.sum((numpy.abs(y_test - yy)))

uu = uu/len(y_test)

print(len(y_test))




print(uu)