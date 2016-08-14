import tensorflow as tf


def gpu_session(graph, mem_frac=0.2):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    return tf.Session(config=tf.ConfigProto(device_count={"GPU": 1},
                                            gpu_options=gpu_options))


def minimize_with_grad_clip(opt, clip_norm, *args, **kwargs):
    grads_and_vars = opt.compute_gradients(*args, **kwargs)
    grads, vars = zip(*grads_and_vars)
    clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm)
    clipped_grads_and_vars = zip(clipped_grads, vars)
    return opt.apply_gradients(clipped_grads_and_vars)


def moments(x):
    '''
    An implementation of tf.nn.moments (along axis [0]) that is more numerically stable,
    and is guaranteed to produce nonnegative variance.
    https://github.com/tensorflow/tensorflow/issues/3290
    '''
    with tf.name_scope('moments'):
        mean = tf.reduce_mean(x)
        shifted = x - mean
        return mean, tf.nn.relu(tf.reduce_mean(shifted ** 2))
