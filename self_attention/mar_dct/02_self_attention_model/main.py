import os
import tensorflow as tf
import numpy as np
from ops import *
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, batch_and_drop_remainder
import tensorflow.contrib.slim as slim
from sklearn.preprocessing import StandardScaler

"""Self-Attention Based Vessel Classification (Matching) Network"""
class SelfAttentionModel:
    def __init__(self, data_X, data_y):
        self.n_class = 35
        self.n_ch = 2048
        #self.n_ch = 1024
        self.sn = True
        self.weight_decay_rate = 0.0002
        self._create_architecture(data_X, data_y)

    def _create_architecture(self, data_X, data_y):
        y_hot = tf.one_hot(data_y, depth = self.n_class)

        self.gt_label = data_y
        self.logits = self._create_model(data_X, self.n_ch, self.sn)
        predictions = tf.nn.softmax(self.logits)
        predictions = tf.argmax(predictions, 1, output_type = tf.int32)
        #self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_hot, 
        #                                                                     logits = logits))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_hot, 
                                                                              logits = self.logits))
        self.loss += self._decay()
        #self.loss = tf.reduce_sum(tf.keras.losses.categorical_hinge(y_hot,logits))
        #self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(self.loss)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001).minimize(self.loss)
        self.optimizer = tf.train.MomentumOptimizer(0.0000095, 0.9).minimize(self.loss)
        # best so far: 117/131, around 150 iterations with only one layer
        #-- the last layer to produce 35 outputs without dropout or anything
        #self.optimizer = tf.train.MomentumOptimizer(0.00001, 0.9).minimize(self.loss)
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions, data_y), tf.float32))

    def _create_model(self, x, channels, sn):

        #x = slim.conv2d(x, 1024, [1, 1], padding = 'VALID')

        #x = relu(x)
        batch_size, height, width, num_channels = x.get_shape().as_list()
        # self-attention
        f = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
        f = max_pooling(f)

        g = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']

        h = conv(x, channels // 2, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]
        h = max_pooling(h)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
        o = conv(o, channels, kernel=1, stride=1, sn=sn, scope='attn_conv')
        x = gamma * o + x

        # additional layers after selt-attention
        #x = slim.conv2d(x, 1024, [7, 7], padding = 'VALID')
        #x = slim.conv2d(x, 4096, [1, 1], padding = 'VALID')

        # fc with flatten
        #x = tf.reshape(x, [-1, 1024]) 
        #x = batch_norm(x, is_training=True, scope='bn_3')
        #x = slim.fully_connected(x, 512)
        #x = slim.fully_connected(x, self.n_class, activation_fn = None)

        x = slim.avg_pool2d(x, [7, 7])
        x = tf.reshape(x, [-1, 2048]) 
        #x = slim.fully_connected(x, 1000)
        #x = slim.fully_connected(x, 256)
        #x = slim.fully_connected(x, self.n_class, activation_fn = None)
        #x = slim.fully_connected(x, self.n_class, activation_fn = tf.nn.softmax)

        # extracting feature map at here
        self.feature_map = x

        with tf.variable_scope('unit_last'):
            #x = batch_norm(x, scope = 'bn1')
            #x = self._fully_connected('fc1', x, 1024)
            #x = relu(x)

            #x = slim.dropout(x, 0.8, scope='dropout')
            #x = batch_norm(x, scope = 'bn2')
            #x = self._fully_connected('fc2', x, 64)
            #x = relu(x)

            #x = slim.dropout(x, 0.8, scope='dropout2')
            #x = batch_norm(x, scope = 'bn3')
            x = self._fully_connected('fc3', x, self.n_class)

        return x

    def _fully_connected(self, name, x, out_dim):
        """FullyConnected layer for final output."""
        with tf.variable_scope(name):
          batch_size = 1
          #batch_size = 131
          x = tf.reshape(x, [batch_size, -1])
          w = tf.get_variable(
              'DW', [x.get_shape()[1], out_dim],
              initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
          b = tf.get_variable('biases', [out_dim],
                              initializer=tf.constant_initializer())
          return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
          if var.op.name.find(r'DW') > 0:
            costs.append(tf.nn.l2_loss(var))
            # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.weight_decay_rate, tf.add_n(costs))

def load_n_test():
    print ("load trained model and run test")
    # load maritime data
    # res5c_features
    # to extract feature maps from training set
    test_features = np.load('/workspace/01_feature_extraction/res5c_features/Y_train.npy').astype(np.float32)
    test_labels = np.load('/workspace/01_feature_extraction/res5c_features/label_train.npy').astype(np.int)

    #test_features = np.load('/workspace/01_feature_extraction/res5c_features/Y_test.npy').astype(np.float32)
    #test_labels = np.load('/workspace/01_feature_extraction/res5c_features/label_test.npy').astype(np.int)

    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert test_features.shape[0] == test_labels.shape[0]

    test_origin_shape = test_features.shape
    test_dataset_num = test_features.shape[0]

    # reshape
    test_features = np.reshape(test_features, (test_origin_shape[0],
                               test_origin_shape[2], test_origin_shape[3], test_origin_shape[4]))

    # transpose
    test_features = np.transpose(test_features, (0,2,3,1))

    # test parameters
    batch_size = 1

    #####################################
    # with initializable iterator

    placeholder_X = tf.placeholder(tf.float32, [None, 7, 7, 2048])
    # pool5 features
    #placeholder_X = tf.placeholder(tf.float32, [None, 1, 1, 2048])
    placeholder_y = tf.placeholder(tf.int32, [None])

    dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_y))
    #dataset = dataset.apply(batch_and_drop_remainder(batch_size))
    dataset = dataset.apply(shuffle_and_repeat(test_dataset_num)).apply(batch_and_drop_remainder(batch_size))

    # create iterator
    iter = dataset.make_initializable_iterator()
    batch_x, batch_y = iter.get_next()
    batch_y = tf.cast(batch_y, tf.int32)

    # our self-attention model
    model = SelfAttentionModel(batch_x, batch_y)

    # open session
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True

    # initialize all variables
    init = tf.global_variables_initializer()

    # saver
    saver = tf.train.Saver()
    checkpoint_dir = './checkpoints/'

    with tf.Session(config=config) as sess:

        # initialize
        sess.run(init)

        # load a checkpoint
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))


        # Initialize iterator with test data
        sess.run(iter.initializer, feed_dict = {placeholder_X: test_features, placeholder_y: test_labels})
        total_batch = int(test_dataset_num/batch_size)

        test_avg_cost = 0.
        test_avg_acc = 0.

        Y_test = []
        label_test = []

        #print (total_batch)
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            #accuracy, c = sess.run([model.accuracy, model.loss])
            accuracy, c, feature_map, gt_label = sess.run([model.accuracy, model.loss, model.feature_map, model.gt_label])

            print (feature_map.shape)
            print (gt_label)
            Y_test.append(feature_map)
            label_test.append(gt_label)

            # Compute average loss and accuracy
            test_avg_cost += c
            test_avg_acc += accuracy

        test_avg_cost /= test_dataset_num
        test_avg_acc /= test_dataset_num

        # show the experimental result
        print("On Test set:", "test_accuracy={:.9f}".format(test_avg_acc), "test_cost={:.9f}".format(test_avg_cost))


def train():
    # load maritime data
    train_features = np.load('/workspace/01_feature_extraction/res5c_features/Y_train.npy').astype(np.float32)
    train_labels = np.load('/workspace/01_feature_extraction/res5c_features/label_train.npy').astype(np.int)
    test_features = np.load('/workspace/01_feature_extraction/res5c_features/Y_test.npy').astype(np.float32)
    test_labels = np.load('/workspace/01_feature_extraction/res5c_features/label_test.npy').astype(np.int)

    """
    train_features = np.load('/workspace/01_feature_extraction/pool5_features/Y_train.npy').astype(np.float32)
    train_labels = np.load('/workspace/01_feature_extraction/pool5_features/label_train.npy').astype(np.int)
    test_features = np.load('/workspace/01_feature_extraction/pool5_features/Y_test.npy').astype(np.float32)
    test_labels = np.load('/workspace/01_feature_extraction/pool5_features/label_test.npy').astype(np.int)
    """

    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert train_features.shape[0] == train_labels.shape[0]

    train_origin_shape = train_features.shape
    test_origin_shape = test_features.shape

    """
    # normalize
    train_features = np.reshape(train_features, (train_origin_shape[0],
                                train_origin_shape[2] * train_origin_shape[3] * train_origin_shape[4]))
    test_features = np.reshape(test_features, (test_origin_shape[0],
                               test_origin_shape[2] * test_origin_shape[3] * test_origin_shape[4]))
    scaler = StandardScaler()
    scaler.fit(train_features)
    test_features = scaler.transform(test_features)

    print (train_features.shape)
    print (train_labels.shape)
    """

    # reshape
    train_features = np.reshape(train_features, (train_origin_shape[0],
                                train_origin_shape[2], train_origin_shape[3], train_origin_shape[4]))
    test_features = np.reshape(test_features, (test_origin_shape[0],
                               test_origin_shape[2], test_origin_shape[3], test_origin_shape[4]))

    # transpose
    train_features = np.transpose(train_features, (0,2,3,1))
    test_features = np.transpose(test_features, (0,2,3,1))

    print (train_features.shape)
    print (train_labels.shape)

    #print (train_labels[0], train_labels[1])
    #print (test_features.shape)
    #print (test_labels.shape)

    # training parameters
    train_dataset_num = train_features.shape[0]
    test_dataset_num = test_features.shape[0]
    #batch_size = 32
    #batch_size = 5
    batch_size = 1
    training_epochs = 500
    display_step = 1

    # need to make as tf record files if the size of the dataset is too big to load into memory
    #####################################
    # with one-shot iterator
    #train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    #test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))

    #train_dataset = train_dataset.apply(shuffle_and_repeat(dataset_num)).apply(batch_and_drop_remainder(batch_size))

    # create iterator
    #iter = train_dataset.make_one_shot_iterator()
    #batch_x, batch_y = iter.get_next()
    #batch_y = tf.cast(batch_y, tf.int32)

    #####################################
    # with initializable iterator

    placeholder_X = tf.placeholder(tf.float32, [None, 7, 7, 2048])
    # pool5 features
    #placeholder_X = tf.placeholder(tf.float32, [None, 1, 1, 2048])
    placeholder_y = tf.placeholder(tf.int32, [None])

    dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_y))
    dataset = dataset.apply(shuffle_and_repeat(train_dataset_num)).apply(batch_and_drop_remainder(batch_size))

    # create iterator
    iter = dataset.make_initializable_iterator()
    batch_x, batch_y = iter.get_next()
    batch_y = tf.cast(batch_y, tf.int32)

    # our self-attention model
    model = SelfAttentionModel(batch_x, batch_y)

    # open session
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True

    # initialize all variables
    init = tf.global_variables_initializer()

    # saver
    saver = tf.train.Saver()
    save_step = 5
    checkpoint_dir = './checkpoints/'
    model_name = 'sa_vessel'

    with tf.Session(config=config) as sess:

        # initialize
        sess.run(init)

        for epoch in range(training_epochs):
            train_avg_cost = 0.
            train_avg_acc = 0.

            test_avg_cost = 0.
            test_avg_acc = 0.

            tt = 0

            # Initialize iterator with training data
            sess.run(iter.initializer, feed_dict = {placeholder_X: train_features, placeholder_y: train_labels})

            total_batch = int(train_dataset_num/batch_size)
            #print (total_batch)
            for i in range(total_batch):
                # Run optimization op (backprop) and cost op (to get loss value)
                accuracy, _, c = sess.run([model.accuracy, model.optimizer, model.loss])
                # Compute average loss and accuracy
                train_avg_cost += c
                train_avg_acc += accuracy

            train_avg_cost /= train_dataset_num
            train_avg_acc /= train_dataset_num

            # Initialize iterator with test data
            sess.run(iter.initializer, feed_dict = {placeholder_X: test_features, placeholder_y: test_labels})
            total_batch = int(test_dataset_num/batch_size)
            #print (total_batch)
            for i in range(total_batch):
                # Run optimization op (backprop) and cost op (to get loss value)
                accuracy, c = sess.run([model.accuracy, model.loss])
                # Compute average loss and accuracy
                test_avg_cost += c
                test_avg_acc += accuracy

                tt += 1

            print (test_avg_acc, tt)

            test_avg_cost /= test_dataset_num
            test_avg_acc /= test_dataset_num

            # display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "training_accuracy={:.9f}".format(train_avg_acc), "train_cost={:.9f}".format(train_avg_cost))
                print("Epoch:", '%04d' % (epoch+1), "test_accuracy={:.9f}".format(test_avg_acc), "test_cost={:.9f}".format(test_avg_cost))

            # save checkpoints
            if epoch % save_step == 0:
                saver.save(sess, os.path.join(checkpoint_dir, model_name+'.model'), global_step=epoch)
                print("Epoch:", '%04d' % (epoch+1), "saving checkpoint")


"""main"""
def main():

    # load the trained model and test it
    load_n_test()

    # train the model
    #train()

if __name__ == '__main__':
    main()
