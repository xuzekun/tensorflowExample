import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def createOneHotLables(lables):
    new_lables = np.zeros((lables.shape[0], 10))
    new_lables[np.arange(lables.shape[0]), lables] = 1
    return new_lables

TRAIN_NUM = 38000
BATCH_SIZE = 2000
LEARNING_RATE = 0.01

trainData = pd.read_csv('../data/kaggle/digitRecognizer/train.csv')
testData = pd.read_csv('../data/kaggle/digitRecognizer/test.csv')

imgs = trainData.iloc[:,1:].values.astype(np.float)
imgs = imgs / 255.0
lables = trainData.iloc[:,0].values


lables = createOneHotLables(lables)
lables = lables.astype(np.float)

testImgs = testData.iloc[:,:].values.astype(np.float)
testImgs = testImgs / 255.0

testNum = len(testImgs)
print(testNum)

#print(imgs[0])
#print(lables.shape)
#print(testImgs.shape)

index = np.random.permutation(imgs.shape[0])

train_idx, val_idx = index[0:TRAIN_NUM], index[TRAIN_NUM:]

trainImgs, valImgs = imgs[train_idx],imgs[val_idx]

trainLables, valLables = lables[train_idx], lables[val_idx]

valNum = len(valImgs)
#print('val')
#print(valNum)

trainDatas = (trainImgs, trainLables)
valDatas = (valImgs, valLables)

trainDataset = tf.data.Dataset.from_tensor_slices(trainDatas)
trainDataset = trainDataset.batch(BATCH_SIZE)

valDataset = tf.data.Dataset.from_tensor_slices(valDatas)
valDataset = valDataset.batch(BATCH_SIZE)

print(trainDataset.output_shapes)
print(valDataset.output_shapes)
print(trainDataset.output_types)

#print(trainDatas)

print('----')
#print(valDatas)

iterator = tf.data.Iterator.from_structure(trainDataset.output_types, trainDataset.output_shapes)
img, label = iterator.get_next()

trainIter = iterator.make_initializer(trainDataset)
valIter = iterator.make_initializer(valDataset)

#  --test
testDataset = tf.data.Dataset.from_tensor_slices(testImgs)
testDataset = testDataset.batch(testNum)
testIterator = tf.data.Iterator.from_structure(testDataset.output_types, testDataset.output_shapes)
testimg = testIterator.get_next()

testIter = testIterator.make_initializer(testDataset)

w = tf.get_variable('weights', shape=(784,10), initializer=tf.random_normal_initializer(0,0.01),dtype=tf.float64)
b = tf.get_variable('bias', shape=(1,10), initializer=tf.zeros_initializer(),dtype=tf.float64)

logits = tf.add(tf.matmul(img, w) ,b)

entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logits)
loss = tf.reduce_mean(entropy)

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

pred = tf.nn.softmax(logits)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
correct_num = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

#
test_pred = tf.nn.softmax(tf.add(tf.matmul(testimg, w) ,b))
test_result = tf.argmax(test_pred, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for i in range(15):
        sess.run(trainIter)
        totalLoss = 0
        try:
            while True:
                _,l = sess.run([optimizer,loss])
                totalLoss += l

        except tf.errors.OutOfRangeError:
            pass

        print('-------%d-------' % i)
        print('total loss: %s' % totalLoss)

        # val
        sess.run(valIter)

        totalCorrectNum = 0
        try:
            while True:
                num = sess.run(correct_num)
                totalCorrectNum += num
        except tf.errors.OutOfRangeError:
            pass

        print('val set correct num:%d' % totalCorrectNum)
        print('val correct: %f' % (float(totalCorrectNum)/ valNum))

    sess.run(testIter)
    result = sess.run(test_result)
    print(result)
    print(len(result))

    saveDataFrame = pd.DataFrame({'ImageId':range(1,testNum+1),'Label':result},columns=['ImageId', 'Label'])
    saveDataFrame.to_csv('123.csv',index=False)