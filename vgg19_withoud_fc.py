import scipy.io
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from scipy.misc import imread

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
    return tf.nn.max_pool(input, ksize=(1,2,2,1),strides=(1,2,2,1), padding="SAME")

def net(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0,1))
    print( mean_pixel)
    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
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

        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = pool_layer(current)

        net[name] = current
    assert  len(net) == len(layers)
    return net,mean_pixel, layers



#Test

vgg_path = './imagenet-vgg-verydeep-19.mat'
img_path = './10.jpg'
input_image = imread(img_path).astype(np.float)
print(input_image.shape)
shape = (1,input_image.shape[0], input_image.shape[1], input_image.shape[2])

with tf.Session() as sess:
    image = tf.placeholder(tf.float32, shape=shape)
    nets, mean_pixel, layers = net(vgg_path, image)
    input_image_pre = np.array([preprocess(input_image, mean_pixel)])
    print("layers")
    print(layers)
    for i,layer in enumerate(layers):
        print("%d / %d  %s" %(i+1,len(layers),layer))
        #features = sess.run(nets[layer],feed_dict={image:input_image_pre})
        features = nets[layer].eval(feed_dict={image: input_image_pre})

        print(" Type of 'features' is ", type(features))
        print(" Shape of 'features' is %s" % (features.shape,))

        plt.figure(i+1,figsize=(10,5))
        plt.matshow(features[0, :, :, 0], cmap=plt.cm.gray, fignum=i+1)
        plt.show()



def test():
    cwd = os.getcwd()
    VGG_PATH = cwd + "./imagenet-vgg-verydeep-19.mat"
    vgg = scipy.io.loadmat(VGG_PATH)
    # 先显示一下数据类型，发现是dict
    print(type(vgg))
    # 字典就可以打印出键值dict_keys(['__header__', '__version__', '__globals__', 'layers', 'classes', 'normalization'])
    print(vgg.keys())
    # 进入layers字段，我们要的权重和偏置参数应该就在这个字段下
    layers = vgg['layers']

    # 打印下layers发现输出一大堆括号，好复杂的样子：[[ array([[ (array([[ array([[[[ ,顶级array有两个[[
    # 所以顶层是两维,每一个维数的元素是array,array内部还有维数
    # print(layers)

    # 输出一下大小，发现是(1, 43)，说明虽然有两维,但是第一维是”虚的”,也就是只有一个元素
    # 根据模型可以知道,这43个元素其实就是对应模型的43层信息(conv1_1,relu,conv1_2…),Vgg-19没有包含Relu和Pool,那么看一层就足以,
    # 而且我们现在得到了一个有用的index,那就是layer,layers[layer]
    print("layers.shape:", layers.shape)
    layer = layers[0]
    # 输出的尾部有dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')])
    # 可以看出顶层的array有5个元素,分别是weight(含有bias), pad(填充元素,无用), type, name, stride信息,
    # 然后继续看一下shape信息,
    print("layer.shape:", layer.shape)
    # print(layer)输出是(1, 1),只有一个元素
    print("layer[0].shape:", layer[0].shape)
    # layer[0][0].shape: (1,),说明只有一个元素
    print("layer[0][0].shape:", layer[0][0].shape)

    # layer[0][0][0].shape: (1,),说明只有一个元素
    print("layer[0][0][0].shape:", layer[0][0][0].shape)
    # len(layer[0][0]):5，即weight(含有bias), pad(填充元素,无用), type, name, stride信息
    print("len(layer[0][0][0]):", len(layer[0][0][0]))
    # 所以应该能按照如下方式拿到信息，比如说name，输出为['conv1_1']
    print("name:", layer[0][0][0][3])
    # 查看一下weights的权重，输出(1,2),再次说明第一维是虚的,weights中包含了weight和bias
    print("layer[0][0][0][0].shape", layer[0][0][0][0].shape)
    print("layer[0][0][0][0].len", len(layer[0][0][0][0]))

    # weights[0].shape: (2,),weights[0].len: 2说明两个元素就是weight和bias
    print("layer[0][0][0][0][0].shape:", layer[0][0][0][0][0].shape)
    print("layer[0][0][0][0].len:", len(layer[0][0][0][0][0]))

    weights = layer[0][0][0][0][0]
    # 解析出weight和bias
    weight, bias = weights
    # weight.shape: (3, 3, 3, 64)
    print("weight.shape:", weight.shape)
    # bias.shape: (1, 64)
    print("bias.shape:", bias.shape)