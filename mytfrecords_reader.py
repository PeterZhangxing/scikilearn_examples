import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("cifar_dir", "data/cifar-10-batches-py/", "文件的目录")
tf.app.flags.DEFINE_string("cifar_tfrecords", "tmp/cifar.tfrecords", "存进tfrecords的文件")


def sess_wrapper(func):
    def inner(self,image_batch,label_batch,*args, **kwargs):
        # print(self)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord=coord)

            func(self,image_batch,label_batch,*args,**kwargs)

            coord.request_stop()
            coord.join(threads)

    return inner


class CifarRead(object):

    def __init__(self,file_dir):
        self.file_dir = file_dir
        self.file_li = self.built_file_li()

        self.height = 32
        self.width = 32
        self.channel = 3

        self.label_bytes = 1
        self.image_bytes = self.height * self.width * self.channel
        self.bytes = self.label_bytes + self.image_bytes

    def built_file_li(self):
        file_names = os.listdir(self.file_dir)
        file_li = [ os.path.join(self.file_dir,name) for name in file_names if name[-1] in [ str(i) for i in range(1,6) ] ]
        return file_li

    def read_and_decode(self):
        file_q = tf.train.string_input_producer(self.file_li)

        reader = tf.FixedLengthRecordReader(self.bytes)
        key,val = reader.read(file_q)

        label_image = tf.decode_raw(bytes=val,out_type=tf.uint8)
        # print(label_image) # Tensor("DecodeRaw:0", shape=(?,), dtype=uint8)

        label = tf.cast(tf.slice(label_image,[0],[self.label_bytes]),tf.int32)
        # print(label)
        image = tf.slice(label_image,[self.label_bytes],[self.image_bytes])
        # print(image)
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        # print(image_reshape) # Tensor("Reshape:0", shape=(32, 32, 3), dtype=uint8)

        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)
        # print(image_batch,label_batch) # Tensor("batch:0", shape=(10, 32, 32, 3), dtype=uint8) Tensor("batch:1", shape=(10, 1), dtype=int32)

        # self.image_batch = image_batch
        # self.label_batch = label_batch
        # with tf.Session() as sess:
        #     coord = tf.train.Coordinator()
        #     threads = tf.train.start_queue_runners(sess, coord=coord)
        #
        #     print(sess.run([image_batch,label_batch]))
        #
        #     coord.request_stop()
        #     coord.join(threads)

        return image_batch, label_batch

    # @sess_wrapper
    def write_ro_tfrecords(self,image_batch, label_batch):
        writer = tf.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)
        for i in range(10):
            image = image_batch[i].eval().tostring()
            label = int(label_batch[i].eval()[0])
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
            writer.write(example.SerializeToString())
        writer.close()

        return None

    def read_from_tfrecords(self):
        file_queue = tf.train.string_input_producer([FLAGS.cifar_tfrecords])

        reader = tf.TFRecordReader()
        key, val = reader.read(file_queue)

        features = tf.parse_single_example(val, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })

        # 4、解码内容, 如果读取的内容格式是string需要解码， 如果是int64,float32不需要解码
        image = tf.decode_raw(features["image"], tf.uint8)

        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        label = tf.cast(features["label"], tf.int32)

        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord=coord)

            print(sess.run([image_batch,label_batch]))

            coord.request_stop()
            coord.join(threads)

        # return image_batch, label_batch
        return None


if __name__ == '__main__':
    cfr = CifarRead(FLAGS.cifar_dir)
    # print(cfr.file_li)
    # cfr.read_and_decode()
    cfr.read_from_tfrecords()

    # image_batch, label_batch = cfr.read_from_tfrecords()
    # # image_batch, label_batch = cfr.read_and_decode()
    #
    # # 开启会话运行结果
    # with tf.Session() as sess:
    #     # 定义一个线程协调器
    #     coord = tf.train.Coordinator()
    #     # 开启读文件的线程
    #     threads = tf.train.start_queue_runners(sess, coord=coord)
    #
    #     # 存进tfrecords文件
    #     # print("开始存储")
    #     # cfr.write_ro_tfrecords(image_batch, label_batch)
    #     # print("结束存储")
    #
    #     # # 打印读取的内容
    #     print(sess.run([image_batch, label_batch]))
    #
    #     # 回收子线程
    #     coord.request_stop()
    #     coord.join(threads)
