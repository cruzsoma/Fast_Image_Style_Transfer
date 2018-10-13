import tensorflow as tf
from preprocessing import preprocessing_factory
import model
import time
import os
from consts import *

style_model_file = "models/style1.ckpt-1400"
content_image = "images/test2.jpg"

def get_image(path, height, width, preprocess_fn):
    png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    return preprocess_fn(image, height, width)

def main(_):
    evaluate(content_image, style_model_file)

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

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()