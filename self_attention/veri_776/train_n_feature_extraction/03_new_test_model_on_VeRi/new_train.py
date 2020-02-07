from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import math

slim = tf.contrib.slim
from ops import *

# dataset Veri-776
import veri_776 
# inception v3 and mobilenet v1 use the same preprocessing procedcure
import inception_preprocessing

# inception v3
from inception_v3 import inception_v3, inception_v3_arg_scope
#image_size = inception_v3.default_image_size
# mobilenet v1
from mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope
#image_size = mobilenet_v1.default_image_size
# mobilenet v1 with self-attention (sa)
from mobilenet_v1_w_sa import mobilenet_v1_w_sa, mobilenet_v1_w_sa_arg_scope
image_size = mobilenet_v1_w_sa.default_image_size


LIB_NAME = 'extra_losses'

def load_op_module(lib_name):
  lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tf.extra_losses/build/lib{0}.so'.format(lib_name))
  oplib = tf.load_op_library(lib_path)
  return oplib

op_module = load_op_module(LIB_NAME)


tf.flags.DEFINE_integer('batch_size', 72, 'Batch size')
tf.flags.DEFINE_integer('epochs', 100, 'Number of training epochs')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate')

tf.flags.DEFINE_string('log_dir', './mobilenet_v1_w_n_cos_loss_batch_sample_logs', 
                        'The directory to save the model files in')
tf.flags.DEFINE_string('dataset_dir', './tfrecords/train',
                        'The directory where the dataset files are stored')
tf.flags.DEFINE_string('checkpoint', './logs',
                        'The directory where the pretrained model is stored')
tf.flags.DEFINE_integer('num_classes', 576,
                        'Number of classes')


FLAGS = tf.app.flags.FLAGS

def get_init_fn(checkpoint_dir):
    checkpoint_exclude_scopes = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(os.path.join(checkpoint_dir, 'inception_v3.ckpt'),
            variables_to_restore)


def main(_):
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # Select the dataset
        # veri-776
        # The below code are heavily from https://github.com/VisualComputingInstitute/triplet-reid
        # Load the data from the CSV file.
        csv_file_path = './VeRi/filenames_w_labels.csv'
        image_root = './VeRi'
        # return type: numpy arrays
        labels, fnames = load_dataset(csv_file_path, image_root, FLAGS.num_classes)
        max_fname_len = max(map(len, fnames))  # We'll need this later for logfiles.
        num_of_samples = len(fnames)

        # Setup a tf.Dataset where one "epoch" loops over all labels.
        # labels are shuffled after every epoch and continue indefinitely.
        unique_labels = np.unique(labels)
        dataset = tf.data.Dataset.from_tensor_slices(unique_labels)
        dataset = dataset.shuffle(len(unique_labels))

        # Constrain the dataset size to a multiple of the batch-size, so that
        # we don't get overlap at the end of each epoch.
        batch_p = 18
        dataset = dataset.take((len(unique_labels) // batch_p) * batch_p)
        dataset = dataset.repeat(None)  # Repeat forever. Funny way of stating it.

        # For every label(identity), get K images.
        batch_k = 4
        dataset = dataset.map(lambda label: sample_k_files_for_id(
        label, all_fnames=fnames, all_labels=labels, batch_k=batch_k))

        # Ungroup/flatten the batches for easy loading of the files.
        dataset = dataset.apply(tf.contrib.data.unbatch())

        # Convert filenames to actual image tensors.
        loading_threads = 4
        net_input_size = (image_size, image_size)
        dataset = dataset.map(
            lambda fname, label: fname_to_image_tensor(
                fname, label, image_root=image_root, image_size=net_input_size),
            num_parallel_calls=loading_threads)

        # Group it back into PK batches.
        batch_size = batch_p * batch_k
        dataset = dataset.batch(batch_size)

        # Overlap producing and consuming for parallelism.
        # prepetch 1 batch
        dataset = dataset.prefetch(1)

        # Since we repeat the data infinitely, we only need a one-shot iterator.
        iter = dataset.make_one_shot_iterator()
        images, fnames, labels = iter.get_next()

        #labels = tf.reshape(labels, [FLAGS.num_classes])


        # mobilenet v1 with self-attention (sa)
        with slim.arg_scope(mobilenet_v1_w_sa_arg_scope()):
            #logits, _ = mobilenet_v1_w_sa(images, num_classes = FLAGS.num_classes, is_training=True)
            logits, _ = mobilenet_v1_w_sa(images, num_classes = None, is_training=True)

        #predictions = tf.nn.softmax(logits, name='prediction')

        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        loss = cos_loss(logits, labels, FLAGS.num_classes, alpha=0.35)

        # Add summaries
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = slim.learning.create_train_op(loss, optimizer)

        num_batches = math.ceil(num_of_samples/float(FLAGS.batch_size)) 
        num_steps = FLAGS.epochs * int(num_batches)
        print('num_steps: {}, num_batches: {}'.format(num_steps,num_batches))

        # open session
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"

        slim.learning.train(
            train_op,
            logdir=FLAGS.log_dir,
            #init_fn=get_init_fn(FLAGS.checkpoint),
            session_config=config,
            number_of_steps=num_steps,
            # save summery every 5 min
            save_summaries_secs=300,
            # save checkpoints every 30 min
            save_interval_secs=3000
        )

        """
        # initialize all variables
        init = tf.global_variables_initializer()

        # saver
        saver = tf.train.Saver()
        checkpoint_dir = './mobilenet_v1_w_sa_cos_loss_batch_sample_logs/'
        save_step = 5000
        model_name = 'mobv1_sa_cos_loss'
        

        with tf.Session(config=config) as sess:
            # initialize
            sess.run(init)
            for epoch in range(num_steps):
                #fname, label, loss_value = sess.run([fnames, labels, loss])
                #print("Epoch: {}, Fname: {}, Label: {}, Loss:{:.4f}".format(i, fname, label, loss_value))
                _, loss_value = sess.run([train_op, loss])
                print("Epoch: {}, Loss: {:.4f}".format(epoch, loss_value))

                # save checkpoints
                if epoch % save_step == 0:
                    saver.save(sess, os.path.join(checkpoint_dir, model_name+'.model'), global_step=epoch)
                    print("Epoch:", '%04d' % (epoch + 1), "saving checkpoint")

                # save last checkpoints
                if epoch % (num_steps) == 0:
                    saver.save(sess, os.path.join(checkpoint_dir, model_name+'.model'), global_step=epoch)
                    print("Epoch:", '%04d' % (epoch + 1), "saving checkpoint")

        """

        print ('done!')

def load_dataset(csv_file, image_root, num_of_classes, fail_on_missing=True):
    """ Loads a dataset .csv file, returning PIDs and FIDs.

    labels are the "vehicle IDs", i.e. class names/labels.
    fnames are the "file names", which are individual relative filenames.

    Args:
        csv_file (string, file-like object): The csv data file to load.
        image_root (string): The path to which the image files as stored in the
            csv file are relative to. Used for verification purposes.
            If this is `None`, no verification at all is made.
        fail_on_missing (bool or None): If one or more files from the dataset
            are not present in the `image_root`, either raise an IOError (if
            True) or remove it from the returned dataset (if False).

    Returns:
        (labels, fnames) a tuple of numpy string arrays corresponding to the PIDs,
        i.e. the identities/classes/labels and the FIDs, i.e. the filenames.

    Raises:
        IOError if any one file is missing and `fail_on_missing` is True.
    """
    dataset = np.genfromtxt(csv_file, delimiter=',', dtype='|U')
    labels, fnames = dataset.T

    # Possibly check if all files exist
    if image_root is not None:
        missing = np.full(len(fnames), False, dtype=bool)
        for i, fname in enumerate(fnames):
            missing[i] = not os.path.isfile(os.path.join(image_root, fname))

        missing_count = np.sum(missing)
        if missing_count > 0:
            if fail_on_missing:
                raise IOError('Using the `{}` file and `{}` as an image root {}/'
                            '{} images are missing'.format(
                                csv_file, image_root, missing_count, len(fnames)))
            else:
                print('[Warning] removing {} missing file(s) from the'
                    ' dataset.'.format(missing_count))
                # We simply remove the missing files.
                fnames = fnames[np.logical_not(missing)]
                labels = labels[np.logical_not(missing)]

    # covert an array of strings to an array fo ints
    labels = labels.astype(np.int)

    """
    # don't need it
    # 1-hot encoding
    num_of_samples = len(labels)
    one_hot_labels = np.zeros((num_of_samples, num_of_classes))
    one_hot_labels[np.arange(num_of_samples), labels] = 1
    #print (num_of_samples, num_of_classes)
    #print (labels)
    #print (one_hot_labels)

    return labels, fnames, one_hot_labels
    """

    return labels, fnames

def sample_k_files_for_id(label, all_fnames, all_labels, batch_k):
    """ Given a label, select K files of that specific label (id). """

    possible_fnames = tf.boolean_mask(all_fnames, tf.equal(all_labels, label))

    # The following simply uses a subset of K of the possible files
    # if more than, or exactly K are available. Otherwise, we first
    # create a padded list of indices which contain a multiple of the
    # original fname count such that all of them will be sampled equally likely.
    count = tf.shape(possible_fnames)[0]
    padded_count = tf.cast(tf.ceil(batch_k / tf.cast(count, tf.float32)), tf.int32) * count
    full_range = tf.mod(tf.range(padded_count), count)

    # Sampling is always performed by shuffling and taking the first k.
    shuffled = tf.random_shuffle(full_range)
    selected_fnames = tf.gather(possible_fnames, shuffled[:batch_k])

    return selected_fnames, tf.fill([batch_k], label)

def fname_to_image_tensor(fname, label, image_root, image_size):
    """ Loads and resizes an image given by FID. Pass-through the PID. """
    # Since there is no symbolic path.join, we just add a '/' to be sure.
    image_encoded = tf.read_file(tf.reduce_join([image_root, '/', fname]))

    # tf.image.decode_image doesn't set the shape, not even the dimensionality,
    # because it potentially loads animated .gif files. Instead, we use either
    # decode_jpeg or decode_png, each of which can decode both.
    # Sounds ridiculous, but is true:
    # https://github.com/tensorflow/tensorflow/issues/9356#issuecomment-309144064
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
    #image_resized = tf.image.resize_images(image_decoded, image_size)

    # Preprocess images
    image_resized = inception_preprocessing.preprocess_image(image_decoded, image_size[0], image_size[1],
            is_training=True)

    return image_resized, fname, label

if __name__ == '__main__':
    tf.app.run()
