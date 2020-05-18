import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("cifar_dir", "./data/cifar10/cifar-10-batches-bin/", "文件的目录")
tf.app.flags.DEFINE_string("cifar_tfrecords", "./tmp/cifar.tfrecords", "存进tfrecords的文件")

def test_sync_read():
    Q = tf.FIFOQueue(3, tf.float32)
    enq_many = Q.enqueue_many([[0.1, 0.2, 0.3], ])
    out_q = Q.dequeue()
    data = out_q + 1
    en_q = Q.enqueue(data)

    with tf.Session() as sess:
        sess.run(enq_many)
        for _ in range(100):
            sess.run(en_q)
        for _ in range(Q.size().eval()):
            print(sess.run(out_q))

    return None

def test_async_read():
    Q = tf.FIFOQueue(1000, tf.float32)
    var = tf.Variable(0.0)
    data = tf.assign_add(var, tf.constant(1.0))
    en_q = Q.enqueue(data)

    qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = qr.create_threads(sess,coord=coord,start=True)

        for _ in range(100):
            print(sess.run(Q.dequeue()))

        coord.request_stop()
        coord.join(threads)

    return None

def test_csv_read():
    data_dir = 'data/test_csv'
    file_names = os.listdir(data_dir)
    file_li = [ os.path.join(data_dir,name) for name in file_names ]

    file_q = tf.train.string_input_producer(file_li)

    reader = tf.TextLineReader()
    key,val = reader.read(file_q)

    # record_defaults:指定每一个样本的每一列的类型，指定默认值[["None"], [4.0]]
    records = [["None"], ["None"]]
    example, label = tf.decode_csv(val, record_defaults=records)
    # example = tf.cast(example,tf.string)

    example_b, label_b = tf.train.batch([example,label],batch_size=9,num_threads=1,capacity=9)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord=coord)

        print(sess.run([example_b, label_b]))

        coord.request_stop()
        coord.join(threads)

    return None

def test_pic_read():
    data_dir = 'data/pics'
    file_names = os.listdir(data_dir)
    file_li = [ os.path.join(data_dir,name) for name in file_names ]
    file_q = tf.train.string_input_producer(file_li)

    reader = tf.WholeFileReader()
    key,val = reader.read(file_q)

    image = tf.image.decode_jpeg(val)
    # print(image) # Tensor("DecodeJpeg:0", shape=(?, ?, ?), dtype=uint8)
    image = tf.image.resize_images(images=image,size=[200,200])
    # print(image) # Tensor("Squeeze:0", shape=(200, 200, ?), dtype=float32)
    image.set_shape([200,200,3])
    # print(image) # Tensor("Squeeze:0", shape=(200, 200, 3), dtype=float32)

    image_batch = tf.train.batch([image], batch_size=3, num_threads=1, capacity=3)
    # print(image_batch) # Tensor("batch:0", shape=(3, 200, 200, 3), dtype=float32)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        print(sess.run(image_batch))

        coord.request_stop()
        coord.join(threads)

    return None


if __name__ == '__main__':
    # test_sync_read()
    # test_async_read()
    # test_csv_read()
    test_pic_read()