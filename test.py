import tensorflow as tf
import numpy as np

r = np.random.randint(0, 10, (10, 5, 4, 3))
print(r[:, :-1, ...].shape)
tf.logging.set_verbosity(tf.logging.INFO)
filename_queue = tf.train.string_input_producer(["B.csv", "A.csv"], shuffle=True, num_epochs=2)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [['null'], ['null']]
data_list = [tf.decode_csv(value, record_defaults=record_defaults) for _ in range(2)]

data_batch, label_batch = tf.train.batch_join(data_list, batch_size=2)


# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(4):
#         # img, label = sess.run([col1, col2])
#         # print(img)
#         # print(label)
#
#         # img, label = sess.run(data_batch)
#         # print(img)
#         # print(label)
#         img, label = sess.run([data_batch, label_batch])
#         print(img)
#         print(label)
#
#     coord.request_stop()
#     coord.join(threads)
#

init_op = tf.initialize_local_variables()
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop():
            img, label = sess.run([data_batch, label_batch])
            print(img)
            print(label)

    except tf.errors.OutOfRangeError:
        tf.logging.info('Done training -- epoch limit reached')
        print("Done")
    finally:
        coord.request_stop()
    coord.join(threads)

