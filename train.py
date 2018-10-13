import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory
from preprocessing import preprocessing_factory
import os
import model
import time
from generate import evaluate

train_image_path = "C:/Github/train2014"
style_image = "img/starry.jpg"
style_name = "style1"
model_path = "models"
loss_model = "vgg_16"
loss_model_file = "pretrained/vgg_16.ckpt"
checkpoint_exclude_scopes = "vgg_16/fc"

content_layers = ["vgg_16/conv3/conv3_3"]
style_layers = ["vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2", "vgg_16/conv3/conv3_3", "vgg_16/conv4/conv4_3"]

content_weight = 1.0
style_weight = 100.0
tv_weight = 0.0

image_size = 256
batch_size = 4
epoch = 2

evaluate_test_image = "img/test3.jpg"

def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, [num_images, -1, num_filters])
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)
    return grams

def get_init_fn(p_loss_model_file, p_checkpoint_exclude_scopes):
    tf.logging.info("Use pretrained model %s" % p_loss_model_file)

    exclusions = []
    if p_checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in p_checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        p_loss_model_file,
        variables_to_restore,
        ignore_missing_vars=True
    )

def get_style_features():
    with tf.Graph().as_default():
        net_fn = nets_factory.get_network_fn(loss_model, num_classes=1, is_training=False)
        preprocessing_fn, unprocessing_fn = preprocessing_factory.get_preprocessing(loss_model, is_training=False)

        image_bytes = tf.read_file(style_image)
        if style_image.lower().endswith("png"):
            image = tf.image.decode_png(image_bytes)
        else:
            image = tf.image.decode_jpeg(image_bytes)

        images = tf.expand_dims(preprocessing_fn(image, image_size, image_size), 0)


        _, endpoint_dict = net_fn(images, spatial_squeeze=False)

        features = []
        for layer in style_layers:
            feature = endpoint_dict[layer]
            feature = tf.squeeze(gram(feature), [0])
            features.append(feature)

        with tf.Session() as sess:
            init_fn = get_init_fn(loss_model_file, checkpoint_exclude_scopes)
            init_fn(sess)

            if os.path.exists("generated") is False:
                os.makedirs("generated")
            save_path = 'generated/' + style_name + '.jpg'
            with open(save_path, 'wb') as f:
                # todo
                target_image = unprocessing_fn(images[0, :])
                image_processed = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(image_processed))
                tf.logging.info('Target style pattern is saved to: %s.' % save_path)

            return sess.run(features)


def calcu_content_loss(dict, layers):
    content_loss = 0
    for layer in layers:
        generated_images, content_images = tf.split(dict[layer], 2, 0)
        size = tf.size(generated_images)
        content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  # remain the same as in the paper

    return content_loss

def calcu_style_loss(dict, style_features, layers):
    style_loss = 0
    style_loss_summary = {}
    for style_feature, layer in zip(style_features, layers):
        generated_images, _ = tf.split(dict[layer], 2, 0)
        size = tf.size(generated_images)
        layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_feature) * 2 / tf.to_float(size)
        style_loss_summary[layer] = layer_style_loss
        style_loss += layer_style_loss
    return style_loss, style_loss_summary

def calcu_total_variation_loss(image):
    shape = tf.shape(image)
    height = shape[1]
    width = shape[2]
    y = tf.slice(image, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(image, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(image, [0, 0, 0, 0], tf.stack([-1, -1, width -1, -1])) - tf.slice(image, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss

def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss


def read_images(path, preprocessing_fn, width, height, batch_size=4, epoch=2, shuffle=True):
    image_names = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if not shuffle:
        image_names = sorted(image_names)

    is_png = image_names[0].lower().endswith('png')
    image_name_queue = tf.train.string_input_producer(image_names, shuffle=shuffle, num_epochs=epoch)
    reader = tf.WholeFileReader()
    _, img_byte = reader.read(image_name_queue)
    image = tf.image.decode_png(img_byte, channels=3) if is_png else tf.image.decode_jpeg(img_byte, channels=3)

    processed_image = preprocessing_fn(image, height, width)
    return tf.train.batch([processed_image], batch_size, dynamic_pad=True)


def main():
    # get the style features of the style image in the loss model
    style_features = get_style_features()

    training_path = os.path.join(model_path, style_name)
    if os.path.exists(training_path) is False:
        os.makedirs(training_path)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            """Build Network"""
            # loss network
            net_fn = nets_factory.get_network_fn(loss_model, num_classes=1, is_training=False)

            # get preprocessing function
            preprocessing_fn, unprocessing_fn = preprocessing_factory.get_preprocessing(loss_model, is_training=False)

            # read images for training
            processed_images = read_images(train_image_path, preprocessing_fn, image_size, image_size, batch_size=batch_size, epoch=2, shuffle=True)

            generated = model.net(processed_images, training=True)
            processed_generated = [preprocessing_fn(image, image_size, image_size) for image in tf.unstack(generated, axis=0, num=batch_size)]
            processed_generated = tf.stack(processed_generated)

            _, endpoints_dict = net_fn(tf.concat([processed_generated, processed_images], 0), spatial_squeeze=False)

            # Log the structure of loss network
            tf.logging.info('Loss network layers(You can define them in "content_layers" and "style_layers"):')
            for key in endpoints_dict:
                tf.logging.info(key)

            # calculate losses
            content_loss = calcu_content_loss(endpoints_dict, content_layers)
            style_loss, style_loss_summary = calcu_style_loss(endpoints_dict, style_features, style_layers)
            tv_loss = total_variation_loss(generated)

            loss = content_loss * content_weight + style_loss * style_weight + tv_loss * tv_weight

            # Add Summary for visualization in tensorboard.
            """Add Summary"""
            tf.summary.scalar('losses/content_loss', content_loss)
            tf.summary.scalar('losses/style_loss', style_loss)
            tf.summary.scalar('losses/regularizer_loss', tv_loss)

            tf.summary.scalar('weighted_losses/weighted_content_loss', content_loss * content_weight)
            tf.summary.scalar('weighted_losses/weighted_style_loss', style_loss * style_weight)
            tf.summary.scalar('weighted_losses/weighted_regularizer_loss', tv_loss * tv_weight)
            tf.summary.scalar('total_loss', loss)

            for layer in style_layers:
                tf.summary.scalar('style_losses/' + layer, style_loss_summary[layer])
            tf.summary.image('generated', generated)
            # tf.image_summary('processed_generated', processed_generated)  # May be better?
            tf.summary.image('origin', tf.stack([
                unprocessing_fn(image) for image in tf.unstack(processed_images, axis=0, num=batch_size)
            ]))
            tf.summary.image('processed', tf.stack([
                image for image in tf.unstack(processed_images, axis=0, num=batch_size)
            ]))
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(training_path, sess.graph)

            """Prepare to Train"""
            global_step = tf.Variable(0, name='global_step', trainable=False)

            variable_to_train = []
            for variable in tf.trainable_variables():
                if not (variable.name.startswith(loss_model)):
                    variable_to_train.append(variable)
            train_optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)

            variables_to_restore = []
            for variable in tf.global_variables():
                if not (variable.name.startswith(loss_model)):
                    variables_to_restore.append(variable)
            saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V1)

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            # Restore variables for loss network.
            init_func = get_init_fn(loss_model_file, checkpoint_exclude_scopes)
            init_func(sess)

            last_file = tf.train.latest_checkpoint(training_path)
            if last_file:
                tf.logging.info('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()
            try:
                while not coord.should_stop():
                    _, loss_t, step = sess.run([train_optimizer, loss, global_step])
                    evaluate_time = time.time() - start_time
                    start_time = time.time()
                    if step % 10 == 0:
                        tf.logging.info('step: %d,  total Loss %f, secs/step: %f' % (step, loss_t, evaluate_time))
                    if step % 25 == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()
                    if step % 200 == 0:
                        saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt'), global_step=step)
                        evaluate(evaluate_test_image, os.path.join(training_path, 'fast-style-model.ckpt' + '-' + str(step)), os.path.join(training_path, str(step) + '.jpg'))
            except tf.errors.OutOfRangeError:
                saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt-done'))
                tf.logging.info('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()