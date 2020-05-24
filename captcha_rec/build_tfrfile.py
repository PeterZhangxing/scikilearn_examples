import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tfrecords_dir',"data/Genpics/tfrecords/cifar.tfrecords",'tf文件存储目录')
tf.app.flags.DEFINE_string('captcha_dir',"data/Genpics/captchas/",'验证图片位置')
tf.app.flags.DEFINE_integer('batch_size',3,'batch_size')

def get_captcha_image():
    filename = []
    for i in range(FLAGS.batch_size):
        string = str(i)+".jpg"
        filename.append(string)

    file_list = [os.path.join(FLAGS.captcha_dir,file) for file in filename]
    file_queue = tf.train.string_input_producer(file_list,shuffle=False)

    reader = tf.WholeFileReader()
    key,val = reader.read(file_queue)

    image = tf.image.decode_jpeg(val)
    image.set_shape([20,80,3])

    image_batch =tf.train.batch([image],batch_size=FLAGS.batch_size,num_threads=1,capacity=FLAGS.batch_size)

    return image_batch

def get_captcha_label():
    file_queue = tf.train.string_input_producer(["data/Genpics/labels.csv"],shuffle=False)

    reader = tf.TextLineReader()
    key,val = reader.read(file_queue)

    records = [[0],["None"]]
    ser,label = tf.decode_csv(val,record_defaults=records)

    label_batch = tf.train.batch([label],batch_size=FLAGS.batch_size,num_threads=1,capacity=FLAGS.batch_size)
    return label_batch

def dealwithlabel(label_str):
    alpa_li = [ chr(i) for i in range(ord("A"),ord("Z")) ]
    index_li = list(range(26))
    alpa_dict = dict(list(zip(alpa_li,index_li)))

    array = []
    for string in label_str:
        letter_list = []
        for letter in string.decode('utf-8'):
            letter_list.append(alpa_dict[letter])
        array.append(letter_list)

    label = tf.constant(array)
    return label

def write_to_tfrecords(image_batch,label_batch):
    label_batch = tf.cast(label_batch,tf.uint8)

    writer = tf.python_io.TFRecordWriter(FLAGS.tfrecords_dir)

    for i in range(FLAGS.batch_size):
        image_string = image_batch[i].eval().tostring()
        label_string = label_batch[i].eval().tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            "image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
            "label":tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_string]))
        }))

        writer.write(example.SerializeToString())

    writer.close()
    return None


if __name__ == '__main__':
    image_batch = get_captcha_image()
    label_batch = get_captcha_label()
    # print(image_batch)
    # print(label)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        label_str = sess.run(label_batch)
        # print(label_str) # [b'SDCV' b'WERF' b'DSAS']
        label_batch = dealwithlabel(label_str)
        # print(label_batch.eval())
        '''
        [[18  3  2 21]
         [22  4 17  5]
         [ 3 18  0 18]]
        '''
        write_to_tfrecords(image_batch,label_batch)

        coord.request_stop()
        coord.join(threads)