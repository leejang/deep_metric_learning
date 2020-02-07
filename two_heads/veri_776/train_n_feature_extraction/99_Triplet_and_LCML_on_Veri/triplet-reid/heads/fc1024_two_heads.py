import tensorflow as tf
from tensorflow.contrib import slim

def head(endpoints, embedding_dim, is_training):
    # head 1
    endpoints['head_output'] = slim.fully_connected(
        endpoints['model_output'], 1024, normalizer_fn=slim.batch_norm,
        normalizer_params={
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        })

    endpoints['emb_raw'] = slim.fully_connected(
        endpoints['head_output'], embedding_dim, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='emb')
    endpoints['emb'] = tf.identity(endpoints['emb_raw'], name="out_emb")
    
    # head 2
    endpoints['head_output_2'] = slim.fully_connected(
        endpoints['model_output'], 1024, normalizer_fn=slim.batch_norm,
        normalizer_params={
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        })

    endpoints['emb_raw_2'] = slim.fully_connected(
        endpoints['head_output_2'], embedding_dim, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='emb_2')
    endpoints['emb_2'] = tf.identity(endpoints['emb_raw_2'], name="out_emb_2")

    return endpoints
