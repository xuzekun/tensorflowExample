import scipy.io
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imresize

def preprocess(image, mean_pixel):
    return image - mean_pixel

def unprocess(image, mean_pixel):
    return image + mean_pixel

def _conv_layer(input, weights, bias):
    print("input")
    print(input)
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1,1,1,1), padding="SAME")
    return tf.nn.bias_add(conv, bias)

def pool_layer(input):
    print(input)
    print("maxpool")
    return tf.nn.max_pool(input, ksize=(1,2,2,1),strides=(1,2,2,1), padding="SAME")

def net(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3','relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3','relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3','relu5_3', 'conv5_4', 'relu5_4', 'pool5',
        'fc6', 'relu6',
        'fc7', 'relu7',
        'fc8', 'softmax',
    )
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0,1))
    print( mean_pixel)
    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        print("current layer: ",current)
        kind = name[:3]
        if kind == 'con':
            kernels, bias = weights[i][0][0][0][0]
            print(kernels.shape)
            # 注意：Mat中的weights参数和tensorflow中不同
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # mat weight.shape: (3, 3, 3, 64)
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            print(kernels.shape)
            bias = bias.reshape(-1)
            print(bias.shape)
            current = _conv_layer(current, kernels, bias)

        elif kind == 'rel':
            current = tf.nn.relu(current)
        elif kind == 'poo':
            current = pool_layer(current)
        elif kind == 'fc6':
            kernels, bias = weights[i][0][0][0][0]
            bias = bias.reshape(-1)
            current = tf.reshape(current,[current.shape[0], -1])
            print("current:",current.shape)
            kernels = kernels.reshape(-1,kernels.shape[-1])
            current = tf.nn.bias_add(tf.matmul(current, kernels),bias)
            print("fc6 kennels: " , kernels.shape)
            print("fc6 bias: ", bias.shape)
        elif kind == 'fc7':
            kernels, bias = weights[i][0][0][0][0]
            bias = bias.reshape(-1)
            print("current:", current.shape)
            kernels = kernels.reshape(-1, kernels.shape[-1])
            current = tf.nn.bias_add(tf.matmul(current, kernels), bias)
            print("fc7 kennels: ", kernels.shape)
            print("fc7 bias: ", bias.shape)
        elif kind == 'fc8':
            kernels, bias = weights[i][0][0][0][0]
            bias = bias.reshape(-1)
            print("current:", current.shape)
            kernels = kernels.reshape(-1, kernels.shape[-1])
            current = tf.nn.bias_add(tf.matmul(current, kernels), bias)
            print("fc8 kennels: ", kernels.shape)
            print("fc8 bias: ", bias.shape)
        elif kind == 'sof':
            current = tf.nn.softmax(current)
        net[name] = current
    assert  len(net) == len(layers)
    return net,mean_pixel, layers



#Test

labels = [str.strip() for str in open("./synset.txt").readlines()]
print(labels)

vgg_path = './imagenet-vgg-verydeep-19.mat'
img_path = './cat2.jpg'
input_image = imread(img_path)
input_image = imresize(input_image, [224,224,3])
input_image = input_image.astype(np.float)

#plt.imshow(input_image)
#plt.show()

print(input_image.shape)
shape = (1,input_image.shape[0], input_image.shape[1], input_image.shape[2])

with tf.Session() as sess:
    image = tf.placeholder(tf.float32, shape=shape)
    nets, mean_pixel, layers = net(vgg_path, image)
    input_image_pre = np.array([preprocess(input_image, mean_pixel)])
    print("layers")
    print(layers)

    result = nets['softmax'].eval(feed_dict={image: input_image_pre})
    print(result)

    # maxIndex = np.argmax(result[0])
    # print("maxIndex:",maxIndex)
    # print(result[0][maxIndex])
    # print(labels[maxIndex])
    import heapq
    index = heapq.nlargest(5,range(len(result[0])),result[0].__getitem__)
    print(index)
    for i in index:
        print("result:" ,result[0][i])
        print(labels[i])
