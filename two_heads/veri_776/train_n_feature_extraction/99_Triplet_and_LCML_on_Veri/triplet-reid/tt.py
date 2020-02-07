import tensorflow as tf
import numpy as np
import math
import numbers

def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2, F).

    Note:
        For convenience, if either `a` or `b` is a `Distribution` object, its
        mean is used.
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)

def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    with tf.name_scope("cdist"):
        diffs = all_diffs(a, b)
        if metric == 'sqeuclidean':
            return tf.reduce_sum(tf.square(diffs), axis=-1)
        elif metric == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
        elif metric == 'cityblock':
            return tf.reduce_sum(tf.abs(diffs), axis=-1)
        elif metric == 'cosine':
            norm_a = tf.nn.l2_normalize(a,1,1e-10)
            norm_b = tf.nn.l2_normalize(b,1,1e-10)
            cos_sim = tf.reduce_sum(tf.multiply(tf.expand_dims(norm_a, axis=1), tf.expand_dims(norm_b, axis=0)), axis=-1)
            dist = 1 - cos_sim
            return cos_sim, dist
            #return tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(x, 0), tf.nn.l2_normalize(y, 0)))
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))
cdist.supported_metrics = [
    'euclidean',
    'sqeuclidean',
    'cityblock',
    'cosine',
]

def py_func(func, inp, Tout, stateful = True, name=None, grad_func=None):
    rand_name = 'PyFuncGrad' + str(np.random.randint(0,1E+8))
    tf.RegisterGradient(rand_name)(grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({'PyFunc':rand_name}):
        return tf.py_func(func,inp,Tout,stateful=stateful, name=name)

def coco_forward(xw, y, m, name=None):
    #pdb.set_trace()
    xw_copy = xw.copy()
    num = len(y)
    orig_ind = range(num)
    xw_copy[orig_ind,y] -= m
    return xw_copy

def coco_help(grad,y):
    grad_copy = grad.copy()
    return grad_copy

def coco_backward(op, grad):
    
    y = op.inputs[1]
    m = op.inputs[2]
    grad_copy = tf.py_func(coco_help,[grad,y],tf.float32)
    return grad_copy,y,m

def coco_func(xw,y,m, name=None):
    with tf.op_scope([xw,y,m],name,"Coco_func") as name:
        coco_out = py_func(coco_forward,[xw,y,m],tf.float32,name=name,grad_func=coco_backward)
        return coco_out

def batch_hard(dists, pids, margin, batch_precision_at_k=None):
    """Computes the batch-hard loss from arxiv.org/abs/1703.07737.

    Args:
        dists (2D tensor): A square all-to-all distance matrix as given by cdist.
        pids (1D tensor): The identities of the entries in `batch`, shape (B,).
            This can be of any type that can be compared, thus also a string.
        margin: The value of the margin if a number, alternatively the string
            'soft' for using the soft-margin formulation, or `None` for not
            using a margin at all.

    Returns:
        A 1D tensor of shape (B,) containing the loss value for each sample.
    """
    with tf.name_scope("batch_hard"):
        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        negative_mask = tf.logical_not(same_identity_mask)
        positive_mask = tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(pids)[0], dtype=tf.bool))

        furthest_positive = tf.reduce_max(dists*tf.cast(positive_mask, tf.float32), axis=1)
        closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])),
                                    (dists, negative_mask), tf.float32)
        # Another way of achieving the same, though more hacky:
        # closest_negative = tf.reduce_min(dists + 1e5*tf.cast(same_identity_mask, tf.float32), axis=1)

        diff = furthest_positive - closest_negative
        if isinstance(margin, numbers.Real):
            diff = tf.maximum(diff + margin, 0.0)
        elif margin == 'soft':
            diff = tf.nn.softplus(diff)
        elif margin.lower() == 'none':
            pass
        else:
            raise NotImplementedError(
                'The margin {} is not implemented in batch_hard'.format(margin))

    if batch_precision_at_k is None:
        return diff, same_identity_mask, negative_mask, positive_mask, furthest_positive, closest_negative 

    # For monitoring, compute the within-batch top-1 accuracy and the
    # within-batch precision-at-k, which is somewhat more expressive.
    with tf.name_scope("monitoring"):
        # This is like argsort along the last axis. Add one to K as we'll
        # drop the diagonal.
        _, indices = tf.nn.top_k(-dists, k=batch_precision_at_k+1)

        # Drop the diagonal (distance to self is always least).
        indices = indices[:,1:]

        # Generate the index indexing into the batch dimension.
        # This is simething like [[0,0,0],[1,1,1],...,[B,B,B]]
        batch_index = tf.tile(
            tf.expand_dims(tf.range(tf.shape(indices)[0]), 1),
            (1, tf.shape(indices)[1]))

        # Stitch the above together with the argsort indices to get the
        # indices of the top-k of each row.
        topk_indices = tf.stack((batch_index, indices), -1)

        # See if the topk belong to the same person as they should, or not.
        topk_is_same = tf.gather_nd(same_identity_mask, topk_indices)

        # All of the above could be reduced to the simpler following if k==1
        #top1_is_same = get_at_indices(same_identity_mask, top_idxs[:,1])

        topk_is_same_f32 = tf.cast(topk_is_same, tf.float32)
        top1 = tf.reduce_mean(topk_is_same_f32[:,0])
        prec_at_k = tf.reduce_mean(topk_is_same_f32)

        # Finally, let's get some more info that can help in debugging while
        # we're at it!
        negative_dists = tf.boolean_mask(dists, negative_mask)
        positive_dists = tf.boolean_mask(dists, positive_mask)

        return diff, top1, prec_at_k, topk_is_same, negative_dists, positive_dists

def cos_loss(x, y,  num_cls, reuse=False, alpha=0.25, scale=64,name = 'cos_loss'):
    '''
    x: B x D - features
    y: B x 1 - labels
    num_cls: 1 - total class number
    alpah: 1 - margin
    scale: 1 - scaling paramter
    '''
    # define the classifier weights
    xs = x.get_shape()
    with tf.variable_scope('centers_var',reuse=reuse) as center_scope:
        w = tf.get_variable("centers", [xs[1], num_cls], dtype=tf.float32, 
            initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
   
    #normalize the feature and weight
    #(N,D)
    x_feat_norm = tf.nn.l2_normalize(x,1,1e-10)
    #(D,C)
    w_feat_norm = tf.nn.l2_normalize(w,0,1e-10)
    
    # get the scores after normalization 
    #(N,C)
    xw_norm = tf.matmul(x_feat_norm, w_feat_norm)  
    #implemented by py_func
    #value = tf.identity(xw)
    #substract the marigin and scale it
    value = coco_func(xw_norm,y,alpha) * scale

    #implemented by tf api
    #margin_xw_norm = xw_norm - alpha
    #label_onehot = tf.one_hot(y,num_cls)
    #value = scale*tf.where(tf.equal(label_onehot,1), margin_xw_norm, xw_norm)

    
    # compute the loss as softmax loss
    cos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=value))

    return cos_loss

def cos_loss_w_bh(x, y,  num_cls, reuse=False, alpha=0.25, scale=64,name = 'cos_loss'):
    '''
    x: B x D - features
    y: B x 1 - labels
    num_cls: 1 - total class number
    alpah: 1 - margin
    scale: 1 - scaling paramter
    B: Batch size, D: Dim of Emb vector, C: num_cls
    '''
    # define the classifier weights
    xs = x.get_shape()
    with tf.variable_scope('centers_var',reuse=reuse) as center_scope:
        w = tf.get_variable("centers", [xs[1], num_cls], dtype=tf.float32, 
            initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
   
    # w: stored center (D, num_cls) => (dim_emb, num_cls)

    #normalize the feature and weight
    #(N,D)
    x_feat_norm = tf.nn.l2_normalize(x,1,1e-10)
    #(D,C)
    w_feat_norm = tf.nn.l2_normalize(w,0,1e-10)
    
    # get the scores after normalization
    # scores: cosine similarity 
    #(N,C)
    xw_norm = tf.matmul(x_feat_norm, w_feat_norm)  

    xw_max = tf.reduce_max(xw_norm, axis=1)

    keeps = tf.where(tf.less(xw_max, 0.7))
    keeps = tf.reshape(keeps, [-1])
    num_active = tf.size(keeps)

    xw_keep = tf.gather(xw_norm, keeps)
    y_keep = tf.gather(y, keeps)

    #value = coco_func(xw_norm,y,alpha) * scale
    value = coco_func(xw_keep,y_keep,alpha) * scale

    return x, y, xw_norm, xw_max, keeps, xw_keep, y_keep, value, num_active

def main():

    print ("Test LMCL with BH")

    # num_cls: 10
    # batch size: 9
    # emb_dim: 5
    num_cls = 10
    features = np.random.rand(9,5).astype(np.float32)
    pids = np.array([1,1,1,2,2,2,3,3,3])

    features = tf.convert_to_tensor(features)
    pids = tf.convert_to_tensor(pids)

    diff, dist = cdist(features, features, metric='cosine')
    #norm_a, norm_b, norm_dist, dist = cdist(features, features, metric='cosine')
    #x, y, xw, xw_max, keeps, xw_keep, y_keep, loss, num_active = cos_loss_w_bh(features, pids, num_cls)

    # open session
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "1"

    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())

      o_diff, o_dist = \
        sess.run([diff, dist])

      #o_x, o_y, o_xw, o_xw_max, o_keeps, o_xw_keep, o_y_keep, o_loss, o_num_active = \
      #  sess.run([x, y, xw, xw_max, keeps, xw_keep, y_keep, loss, num_active])

    #print (o_norm_a)
    #print (o_norm_b)
    #print (o_norm_dist)
    print (o_diff)
    print (o_dist)

    """
    print (o_x)
    print (o_y)
    print (o_xw)
    print (o_xw_max)
    print (o_keeps)
    print (o_xw_keep)
    print (o_y_keep)
    print (o_loss)
    print (o_num_active)
    """

"""
def main():

    print ("Test Triplet Loss with BH")

    # batch size: 9
    # emb_dim: 5
    features = np.random.rand(9,5).astype(np.float32)
    pids = np.array([1,1,1,2,2,2,3,3,3])

    #print (features)
    #print (pids)
 
    dists = cdist(features, features)
    losses, same_id_masks, neg_masks, pos_masks, fur_pos, clo_neg = \
      batch_hard(dists, pids, margin='none', batch_precision_at_k=None)
      #batch_hard(dists, pids, margin='soft', batch_precision_at_k=None)

    # open session
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "1"

    with tf.Session(config=config) as sess:
      o_dist, o_losses, o_same_id_masks, o_neg_masks, o_pos_masks, o_fur_pos, o_clo_neg = \
        sess.run([dists, losses, same_id_masks, neg_masks, pos_masks, fur_pos, clo_neg])

    print (o_dist)
    print (o_losses)
    print (o_same_id_masks)
    print (o_neg_masks)
    print (o_pos_masks)
    print (o_fur_pos)
    print (o_clo_neg)
"""

if __name__ == '__main__':
    main()
