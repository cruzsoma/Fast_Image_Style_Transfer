import tensorflow as tf
from preprocessing import preprocessing_factory
import model
import time
import os
from consts import *
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import numpy as np

style_model_file = "models/metalgear.ckpt-5800"
content_image = "images/test2.jpg"
pb_file_path = "models"
output_height = 256
output_width = 256

def get_image(path, height, width, preprocess_fn):
    png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    return preprocess_fn(image, height, width)

def read_pb_model():
    height = 0
    width = 0
    test_img = []
    with open(content_image, 'rb') as img:
        with tf.Session().as_default() as sess:
            if content_image.lower().endswith('png'):
                test_img = sess.run(tf.image.decode_png(img.read()))
            else:
                test_img = sess.run(tf.image.decode_jpeg(img.read()))
            height = test_img.shape[0]
            width = test_img.shape[1]

    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Session() as sess:
        with gfile.FastGFile(pb_file_path + '/test_model.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        # init
        sess.run(tf.global_variables_initializer())

        input_img = sess.graph.get_tensor_by_name('input_image:0')
        generated_img = sess.graph.get_tensor_by_name('generated_image:0')
        generated_img = tf.cast(generated_img, tf.uint8)
        with open(pb_file_path + '/test_read_model_img.jpg', 'wb') as img:
            # feed_dict = {input_img: test_img, input_h: height, input_w: width}
            feed_dict = {input_img: test_img}
            img.write(sess.run(tf.image.encode_jpeg(generated_img), feed_dict))


def save_pb_model():
    height = 0
    width = 0
    with open(content_image, 'rb') as img:
        with tf.Session().as_default() as sess:
            if content_image.lower().endswith('png'):
                test_img = sess.run(tf.image.decode_png(img.read()))
            else:
                test_img = sess.run(tf.image.decode_jpeg(img.read()))
            height = test_img.shape[0]
            width = test_img.shape[1]

    tf.logging.info('Image size: %dx%d' % (width, height))

    # png = content_image.lower().endswith('png')
    # img_bytes = tf.read_file(content_image)
    # test_img = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)

    with tf.Session(graph=tf.Graph()) as sess:
        input_img = tf.placeholder(tf.float32, [None, None, 3], name="input_image")

        # input_h = tf.placeholder(tf.int32, name="input_height")
        # input_w = tf.placeholder(tf.int32, name="input_width")

        # read image data
        image_processing_fn, _ = preprocessing_factory.get_preprocessing(loss_model, is_training=False)
        image = image_processing_fn(input_img, output_height, output_width)

        image = tf.expand_dims(image, 0)

        generated = model.net(image, training=False)

        generated = tf.squeeze(generated, [0], name='generated_image')

        generated_uint8 = tf.cast(generated, tf.uint8)

        # restore variables
        saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        model_abs_path = os.path.abspath(style_model_file)
        saver.restore(sess, model_abs_path)

        with open(pb_file_path + '/test_model_img.jpg', 'wb') as img:
            # feed_dict = {input_img: test_img, input_h: height, input_w: width}
            feed_dict = {input_img: test_img}
            img.write(sess.run(tf.image.encode_jpeg(generated_uint8), feed_dict))

            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['generated_image'])

            # save model to pb file
            with tf.gfile.FastGFile(pb_file_path + '/test_model.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())


def evaluate(image_file, model_file, path=''):
    height = 0
    width = 0
    with open(image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]

    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:
            # read image data
            image_processing_fn, _ = preprocessing_factory.get_preprocessing(loss_model, is_training=False)
            image = get_image(image_file, height, width, image_processing_fn)

            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            generated = tf.squeeze(generated, [0])

            #restore variables
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            model_abs_path = os.path.abspath(model_file)
            saver.restore(sess, model_abs_path)

            if path:
                generated_image = path
            else:
                generated_image = 'generated/result.jpg'

            if os.path.exists('generated') is False:
                os.makedirs('generated')

            with open(generated_image, 'wb') as img:
                start_time = time.time()
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                end_time = time.time()

                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                tf.logging.info('Done. Please check %s.' % generated_image)

def main(_):
    save_pb_model()
    read_pb_model()
    # evaluate(content_image, style_model_file)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()