import os
import random
import tensorflow as tf
from PIL import Image

FILE_DIR = 'G:\\myGitCode\\tf21\\data\\train\\'
TEST_DIR = 'G:\\myGitCode\\tf21\\data\\test\\'
TRA_RECORD_DIR = 'G:\\myGitCode\\tf21\\data\\train.tfrecords'
VAL_RECORD_DIR = 'G:\\myGitCode\\tf21\\data\\val.tfrecords'
TEST_RECORD_DIR = 'G:\\myGitCode\\tf21\\data\\test.tfrecords'

RATIO = 0.2

def create_filelist(filedir, ratio):
    filelist = []
    for file in os.listdir(FILE_DIR):
        filelist.append(file)

    random.shuffle(filelist)
    sample_size = len(filelist)
    print("sample size is ",sample_size)

    val_size = int(sample_size * ratio)
    train_size = sample_size - val_size
    print("train size is ",train_size)
    print("val size is ", val_size)

    return filelist[:train_size], filelist[train_size:]

def create_record(filelist, record_dir):
    writer = tf.python_io.TFRecordWriter(record_dir)
    for file in filelist:
        name = file.split(sep='.')
        if name[0] == 'cat':
            label = 0
        else:
            label = 1
        img_path = FILE_DIR + file
        img = Image.open(img_path)
        img = img.resize((224,224))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'],tf.uint8)
    img = tf.reshape(img, [224,224,3])
    label = tf.cast(features['label'],tf.int32)
    return img,label

def create_test_record():
    test_images = [TEST_DIR + "%d.jpg" % i for i in range(1, 12501)]
    writer = tf.python_io.TFRecordWriter(TEST_RECORD_DIR)
    for file in test_images:
        img = Image.open(file)
        img = img.resize((224, 224))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

def read_test_record():
    filename_queue = tf.train.string_input_producer([TEST_RECORD_DIR])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'],tf.uint8)
    img = tf.reshape(img, [224,224,3])
    return img

def test(record_dir):
    img, label = read_and_decode(record_dir)

    # 使用shuffle_batch可以随机打乱输入
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=3, capacity=2000, min_after_dequeue=1000)

   # img_batch = tf.train.batch([img, label], batch_size=1, capacity=2000)

    init = tf.global_variables_initializer()

    import matplotlib.pyplot as plt
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        val = None
        for i in range(3):
            val,L = sess.run([img_batch, label_batch])
            print(val.shape)
            # print(val)
            plt.imshow(val[0])
            plt.show()


if __name__ == '__main__':
    #train_list, val_list = create_filelist(FILE_DIR, RATIO)
   # create_record(train_list,TRA_RECORD_DIR)
   # create_record(val_list,VAL_RECORD_DIR)

    #test(TRA_RECORD_DIR)
   # create_test_record()


    img = read_test_record()

    img_batch = tf.train.batch([img], batch_size=3, capacity=2000)

    init = tf.global_variables_initializer()

    import matplotlib.pyplot as plt

    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        val = None
        for i in range(3):
            val = sess.run(img_batch)
            print(val.shape)
            # print(val)
            plt.imshow(val[0])
            plt.show()

