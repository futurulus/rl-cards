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


def add_summary_ops():
    '''
    Connect a `histogram_summary` or `scalar_summary` to every floating point
    tensor. `scalar_summary` operations are added for each scalar `half`, `float`,
    or `double` tensor in the graph, and `histogram_summary` operations for each
    tensor with rank at least 1.

    For all ops in the graph, the `scalar_summary` op for all of its (`half`,
    `float`, or `double`) inputs is guaranteed to run before the `scalar_summary`
    op on any of its outputs.

    Based on `tf.add_check_numerics_ops`.

    Returns:
    A `group` op depending on all `scalar_summary` ops added.
    '''
    summary_op = []
    # This code relies on the ordering of ops in get_operations().
    # The producer of a tensor always comes before that tensor's consumer in
    # this list. This is true because get_operations() returns ops in the order
    # added, and an op can only be added after its inputs are added.
    for op in tf.get_default_graph().get_operations():
        for output in op.outputs:
            if output.dtype in [tf.float16, tf.float32, tf.float64]:
                if output.get_shape().ndims == 0:
                    summ_type = tf.scalar_summary
                else:
                    summ_type = tf.histogram_summary
                message = 'activation/' + op.name + ":" + str(output.value_index)
                with tf.control_dependencies(summary_op):
                    summary_op = [summ_type(message, output)]
    return tf.group(*summary_op)