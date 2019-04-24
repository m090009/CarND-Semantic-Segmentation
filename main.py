#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
from datetime import timedelta



STDDEV = 0.01
L2_REG = 1e-3
LR = 1e-4
KEEP_PROB = 0.85
# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    default_graph = tf.get_default_graph()

    vgg_input_tensor = default_graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = default_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = default_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = default_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = default_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor
print("Load_vgg test:")
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #Skip layers 

    # O1 - 1 
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out,
                                       num_classes, 1, strides=1,
                                       padding= "same",
                                       kernel_initializer= tf.random_normal_initializer(stddev=STDDEV),
                                       kernel_regularizer= tf.contrib.layers.l2_regularizer(L2_REG))
    upsampling_1 = tf.layers.conv2d_transpose(layer7_1x1,
                                              num_classes,
                                              4, 
                                              strides= (2, 2),
                                              padding="same",
                                              kernel_initializer= tf.random_normal_initializer(stddev=STDDEV),
                                              kernel_regularizer= tf.contrib.layers.l2_regularizer(L2_REG))
    
    # Scale the layer 4 and layer 3 pooling layers
    scaled_layer3 = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')
    scaled_layer4 = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')

    layer4_1x1 = tf.layers.conv2d(scaled_layer4,
                                       num_classes, 1, strides=1,
                                       padding= "same",
                                       kernel_initializer= tf.random_normal_initializer(stddev=STDDEV),
                                       kernel_regularizer= tf.contrib.layers.l2_regularizer(L2_REG))
    
    skip_layer_1 = tf.add(upsampling_1, layer4_1x1)


    upsampling_2 = tf.layers.conv2d_transpose(skip_layer_1,
                                              num_classes,
                                              4, 
                                              strides= (2,2),
                                              padding="same",
                                              kernel_initializer= tf.random_normal_initializer(stddev=STDDEV),
                                              kernel_regularizer= tf.contrib.layers.l2_regularizer(L2_REG))

    layer3_1x1 = tf.layers.conv2d(scaled_layer3,
                                    num_classes, 1, strides=1,
                                    padding= "same",
                                    kernel_initializer= tf.random_normal_initializer(stddev=STDDEV),
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(L2_REG))

    skip_layer_2 = tf.add(upsampling_2, layer3_1x1)

    output = tf.layers.conv2d_transpose(skip_layer_2,
                                              num_classes,
                                              16, 
                                              strides= (8, 8),
                                              padding="same",
                                              kernel_initializer= tf.random_normal_initializer(stddev=STDDEV),
                                              kernel_regularizer= tf.contrib.layers.l2_regularizer(L2_REG))
    return output
print("Layers test:")
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Use the L2 loss that was added to the decoder layers in the loss calculation 
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # Some constant weight for the regularization loss
    l2_const = 0.003 # 0.002, 0.005, 0.05, 1
    loss = cross_entropy_loss + l2_const * sum(reg_losses)

    train_op = optimizer.minimize(loss)

    return logits, train_op, cross_entropy_loss
print("Optimize test:")
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    # Training loop 
    losses = []
    print("***************************************************************************")
    print("Start Training for {} epochs with {} batch size ".format(epochs, batch_size))
    for epoch in range(epochs):
        loss = None
        start_time = time.time()
        for image, labels in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image,
                                          correct_label: labels,
                                          keep_prob: KEEP_PROB,
                                          learning_rate: LR})
            losses.append(loss)
        print("Epoch: {} of {}, Loss: {} in {} Time".format(epoch + 1, epochs, round(loss, 4), str(timedelta(seconds=time.time() - start_time))))
    helper.plot_loss('./runs', losses, "Training Loss")
    print("========Training has ended========")
    print("***************************************************************************")
print("Train_nn test:")
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    # helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    

    # Hyperparameters
    # After trying multiple Epohs 200 seems to give ok results 
    EPOCHS = 200
    BATCH_SIZE = 32


    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network



        # Initialize Placeholders
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        # TODO: Train NN using the train_nn function

        # Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)
        s_time = time.time()
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)
        print("Time taken : {}".format(str(timedelta(seconds=(time.time() - s_time)))))
        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
