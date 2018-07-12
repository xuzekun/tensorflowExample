import os
import numpy as np
import tensorflow as tf
TRAIN_DIR = "G:\\myGitCode\\tf21\\data\\train\\"
TEST_DIR = "G:\\myGitCode\\tf21\\data\\test\\"
log_dir = './log/'

IMAGE_SIZE = 208
RATIO = 0.2
N_CLASS = 2
BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 10000
LEARNING_RATE = 0.0001

def get_files(file_dir, ratio):
    cats = []
    label_cats = []
    dogs  = []
    label_dogs = []
    for filename in os.listdir(file_dir):
        if 'cat' in filename:
            cats.append(file_dir + filename)
            label_cats.append(0)
        else:
            dogs.append(file_dir + filename)
            label_dogs.append(1)
    print("There ar %d cats and %d dogs" %(len(cats),len(dogs)))

    # 横向拼接
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    tmp = np.array([image_list, label_list])
    tmp = tmp.transpose()  # 转置
    np.random.shuffle(tmp) # 打乱顺序

    image_list = tmp[:,0]
    label_list = tmp[:,1]

    sample_size = len(label_list)
    val_size = int(np.ceil(sample_size * ratio))
    train_size = sample_size - val_size
    print("Total size: %d, train_size: %d, val_size: %d" % (sample_size, train_size, val_size))

    train_images = image_list[:train_size]
    train_labels = [int(i) for i in label_list[:train_size] ]
    val_images = image_list[train_size:]
    val_labels = [int(i) for i in label_list[train_size:] ]
    return train_images, train_labels, val_images, val_labels

def get_batch(image, label,image_size, batch_size, capacity):
    """
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    """
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    # 构建队列
    input_queue = tf.train.slice_input_producer([image,label])
    image, label = input_queue[0],input_queue[1]
    # 读取图片
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3) # jpg解码
    # 裁剪
    image = tf.image.resize_image_with_crop_or_pad(image,image_size,image_size)
    #image = tf.image.resize_images(image,[image_size, image_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # 标准化 均值为零
    image = tf.image.per_image_standardization(image)

    image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,capacity=capacity,num_threads=64)

    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch

def get_test_batch(batch_size,capacity):
    test_images = [TEST_DIR + "%d.jpg" % i for i in range(1,12501)]
    test_images = np.array(test_images)

    test_images = tf.cast(test_images, tf.string)

    input_queue = tf.train.slice_input_producer([test_images],shuffle=False)
    test_images = input_queue[0]

    image_contents = tf.read_file(test_images)
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    image = tf.image.per_image_standardization(image)

    image_batch = tf.train.batch([image], batch_size=batch_size,capacity=capacity,num_threads=1)

    return image_batch

# # test
# BATCH_SIZE = 5
# CAPACITY = 256
# train_images, train_labels, val_images, val_labels = get_files(TRAIN_DIR,0.2)
# image_batch, label_batch = get_batch(train_images,train_labels, IMAGE_SIZE, BATCH_SIZE, CAPACITY)
#
# import matplotlib.pyplot as plt
#
# with tf.Session() as sess:
#     # 监控队列
#     coord = tf.train.Coordinator()
#     thread = tf.train.start_queue_runners(coord=coord)
#     i = 0
#     try:
#         while not coord.should_stop() and i<1:
#             image, label = sess.run([image_batch,label_batch])
#             print(image)
#             print(image_batch.shape)
#             print(label)
#             print(label_batch)
#             for i in range(BATCH_SIZE):
#                 plt.imshow(image[i])
#                 plt.show()
#             i += 1
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#     coord.join(thread)

def inference(images, batch_size,n_classes):
    """
    Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    """
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable("weights", shape=[3,3,3,16], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", shape=[16],dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation)

        pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
        norm1 = tf.nn.lrn(pool1,depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable("weights", shape=[3,3,16,16], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", shape=[16], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation)
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")


    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',shape=[dim,128],dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases", shape=[128], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases)

    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    with tf.variable_scope('local5') as scope:
        weights = tf.get_variable('weights',shape=[128, n_classes], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[n_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local5 = tf.matmul(local4, weights) + biases

    return local5

def losses(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
    loss = tf.reduce_mean(cross_entropy)
    return loss

def training(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits,labels,1)
    correct = tf.cast(correct, tf.float16)
    accuracy = tf.reduce_mean(correct)
    return  accuracy

def train():
    train_image, train_label, val_image, val_label = get_files(TRAIN_DIR, RATIO)
    train_image_batch, train_label_batch = get_batch(train_image,train_label,IMAGE_SIZE,BATCH_SIZE,CAPACITY)
    val_image_batch, val_label_batch = get_batch(val_image,val_label,IMAGE_SIZE,BATCH_SIZE,CAPACITY)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    logits = inference(x, BATCH_SIZE, N_CLASS)
    loss = losses(logits,y_)
    train_op = training(loss,LEARNING_RATE)
    acc = evaluation(logits,y_)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        try:
            for step in range(MAX_STEP):
                if coord.should_stop():
                    break
                tra_img, tra_label = sess.run([train_image_batch,train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op,loss,acc],feed_dict={x:tra_img,y_:tra_label})
                if step % 50 == 0:
                    print("step: %d,train_loss: %.2f,train_accuracy: %.2f" % (step,tra_loss,tra_acc))
                if step % 200 == 0 or (step+1) == MAX_STEP:
                    val_img, val_label = sess.run([val_image_batch,val_label_batch])
                    val_loss, val_acc = sess.run([loss,acc],feed_dict={x:val_img,y_:val_label})
                    print("** step: %d,val loss= %.2f, val acc: %.2f **" % (step,val_loss,val_acc))
                if step % 2000 == 0 or (step+1) == MAX_STEP:
                    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess,checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print("done training")
        finally:
            coord.request_stop()
        coord.join(threads)

def test():
    TEST_BATCH_SIZE = 100
    test_img_batch = get_test_batch(TEST_BATCH_SIZE, CAPACITY)
    x = tf.placeholder(tf.float32, shape=[TEST_BATCH_SIZE,IMAGE_SIZE, IMAGE_SIZE, 3])
    logit = inference( x,TEST_BATCH_SIZE,N_CLASS)
    logit = tf.nn.softmax(logit)
    saver = tf.train.Saver()

    result = np.zeros(12500)
    index = 0
    import matplotlib.pyplot as plt
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(coord=coord)

        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
        try:
            for i in range(int(12500//TEST_BATCH_SIZE)):
                img = sess.run(test_img_batch)
                out = sess.run(logit,feed_dict={x:img})

                for j in out:
                    result[index] = j[1]
                    index += 1
                print(result[i*TEST_BATCH_SIZE:(i+1)*TEST_BATCH_SIZE])

        except tf.errors.OutOfRangeError:
            print("done training")
        finally:
            coord.request_stop()

        print(result)
        print(len(result))

        import pandas as pd
        saveDataFrame = pd.DataFrame({'id': range(1, result.shape[0] + 1), 'label': result},
                                     columns=['id', 'label'])
        saveDataFrame.to_csv('0709.csv', index=False)

    coord.join(thread)

if __name__ == '__main__':
    #train()
    test()