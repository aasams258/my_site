#!/Users/arthur/Documents/Coding/vgg/venv/bin/python
import numpy as np
import tensorflow as tf
import class_names as classes
from scipy.misc import imread, imresize

# Looks like VGG - D 
'''
0 conv1_1_W (3, 3, 3, 64)
1 conv1_1_b (64,)
2 conv1_2_W (3, 3, 64, 64)
3 conv1_2_b (64,)
4 conv2_1_W (3, 3, 64, 128)
5 conv2_1_b (128,)
6 conv2_2_W (3, 3, 128, 128)
7 conv2_2_b (128,)
8 conv3_1_W (3, 3, 128, 256)
9 conv3_1_b (256,)
10 conv3_2_W (3, 3, 256, 256)
11 conv3_2_b (256,)
12 conv3_3_W (3, 3, 256, 256)
13 conv3_3_b (256,)
14 conv4_1_W (3, 3, 256, 512)
15 conv4_1_b (512,)
16 conv4_2_W (3, 3, 512, 512)
17 conv4_2_b (512,)
18 conv4_3_W (3, 3, 512, 512)
19 conv4_3_b (512,)
20 conv5_1_W (3, 3, 512, 512)
21 conv5_1_b (512,)
22 conv5_2_W (3, 3, 512, 512)
23 conv5_2_b (512,)
24 conv5_3_W (3, 3, 512, 512)
25 conv5_3_b (512,)
26 fc6_W (25088, 4096)
27 fc6_b (4096,)
28 fc7_W (4096, 4096)
29 fc7_b (4096,)
30 fc8_W (4096, 1000)
31 fc8_b (1000,)
'''
# VGG-D model from https://arxiv.org/pdf/1409.1556.pdf.
# Weights from https://www.cs.toronto.edu/~frossard/post/vgg16/#weights
def vgg_model(features, labels, mode):
    weights = np.load("vgg16_weights.npz")
    # Make sure the mean RGB value is subtracted from the pixels.
    input_layer = tf.reshape(features["img"], [-1, 224, 224, 3])
    # Naming scheme: conv<receptive field size>-<num channels>-<layer num>
    conv_1 = tf.layers.conv2d(name="conv3-64-1", inputs=input_layer, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv1_1_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv1_1_b"]))
    conv_2 = tf.layers.conv2d(name="conv3-64-2", inputs=conv_1, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv1_2_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv1_2_b"]))
    pool1 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=[2, 2], strides=2, padding="same")

    conv_3 = tf.layers.conv2d(name="conv3-128-1", inputs=pool1, filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv2_1_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv2_1_b"]))
    conv_4 = tf.layers.conv2d(name="conv3-128-2", inputs=conv_3, filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv2_2_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv2_2_b"]))
    pool2 = tf.layers.max_pooling2d(inputs=conv_4, pool_size=[2, 2], strides=2, padding="same")

    conv_5 = tf.layers.conv2d(name="conv3-256-1", inputs=pool2, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv3_1_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv3_1_b"]))
    conv_6 = tf.layers.conv2d(name="conv3-256-2", inputs=conv_5, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv3_2_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv3_2_b"]))
    conv_7 = tf.layers.conv2d(name="conv3-256-3", inputs=conv_6, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv3_3_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv3_3_b"]))
    pool3 = tf.layers.max_pooling2d(inputs=conv_7, pool_size=[2, 2], strides=2, padding="same")

    conv_8 = tf.layers.conv2d(name="conv3-512-1", inputs=pool3, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv4_1_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv4_1_b"]))
    conv_9 = tf.layers.conv2d(name="conv3-512-2", inputs=conv_8, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv4_2_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv4_2_b"]))
    conv_10 = tf.layers.conv2d(name="conv3-512-3", inputs=conv_9, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv4_3_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv4_3_b"]))
    pool4 = tf.layers.max_pooling2d(inputs=conv_10, pool_size=[2, 2], strides=2, padding="same")

    conv_11 = tf.layers.conv2d(name="conv3-512-4", inputs=pool4, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv5_1_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv5_1_b"]))
    conv_12 = tf.layers.conv2d(name="conv3-512-5", inputs=conv_11, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv5_2_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv5_2_b"]))
    conv_13 = tf.layers.conv2d(name="conv3-512-6", inputs=conv_12, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv5_3_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv5_3_b"]))
    pool5 = tf.layers.max_pooling2d(inputs=conv_13, pool_size=[2, 2], strides=2, padding="same")
    flatten = tf.layers.flatten(inputs=pool5)
    fc1 = tf.layers.dense(inputs=flatten, units=4096, kernel_initializer=tf.constant_initializer(weights["fc6_W"]),
                              bias_initializer=tf.constant_initializer(weights["fc6_b"]))
    fc2 = tf.layers.dense(inputs=fc1, units=4096, kernel_initializer=tf.constant_initializer(weights["fc7_W"]),
                              bias_initializer=tf.constant_initializer(weights["fc7_b"]))
    logits = tf.layers.dense(inputs=fc2, units=1000,kernel_initializer=tf.constant_initializer(weights["fc8_W"]),
                              bias_initializer=tf.constant_initializer(weights["fc8_b"]))

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


# I am passing a numpy input, i need to pass a TF input.
# def predict_fn():
#     filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("laska.png"))
#     image_reader = tf.WholeFileReader()
#     _, image_file = image_reader.read(filename_queue)
#     image = tf.image.decode_png(image_file, channels=3)
#     resized_image = tf.image.resize_images(image, [244, 244])
#     print "pred"
#     tf.Print(resized_image, [resized_image])
#     return tf.estimator.inputs.numpy_input_fn(
#       x={"img": resized_image},
#       shuffle=False)

def predict_fn():
    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))
    # Broad cast remove the mean:
    img = img1 - [123.68, 116.779, 103.939]
    # Zero mean the image
    #mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    return tf.estimator.inputs.numpy_input_fn(
      x={"img": img},
      shuffle=False)

def main():
    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224)) # I think it might need to be 1,224,224(3?)
    # Broad cast remove the mean:
    mimg = img1 - [123.68, 116.779, 103.939]
    img = mimg.reshape(1, 224, 224, 3)
    # Zero mean the image
    #mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"img": img},
      shuffle=False)

    classifier = tf.estimator.Estimator(model_fn=vgg_model, model_dir="model_tmp")
    vals = classifier.predict(input_fn=input_fn)

    print next(vals)
    print "Done"
    #sess = tf.InteractiveSession()

if __name__ == '__main__':
    main()
    # weights = np.load("vgg16_weights.npz")
    # keys = sorted(weights.keys())
    # for k,v in enumerate(keys):
    #     print 86016