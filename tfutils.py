import re
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


NO_MONITOR = [
    r'^CheckNumerics(_\d+)?$',
    r'^Const(_\d+)?$',
    r'^global_norm(_\d+)?/pack(_\d+)?$',
    r'^zeros(_\d+)?$'
]


def add_summary_ops():
    '''
    Connect a `histogram` or `scalar` summary node to every floating point
    tensor. `scalar` summary operations are added for each scalar `half`, `float`,
    or `double` tensor in the graph, and `histogram` summary operations for each
    tensor with rank at least 1. Operations which have names indicating they
    aren't useful to monitor (constants and `CheckNumerics` ops) are skipped.

    For all ops in the graph, the `scalar` summary op for all of its (`half`,
    `float`, or `double`) inputs is guaranteed to run before the `scalar` summary
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
        if op.name and any(re.search(pattern, op.name) for pattern in NO_MONITOR):
            continue
        for output in op.outputs:
            if output.dtype in [tf.float16, tf.float32, tf.float64] and \
                    output.op._get_control_flow_context() == \
                    tf.get_default_graph()._get_control_flow_context():
                if output.get_shape().ndims == 0:
                    summ_type = tf.summary.scalar
                else:
                    summ_type = tf.summary.histogram
                message = op.name + ":" + str(output.value_index)
                with tf.control_dependencies(summary_op):
                    summary_op = [summ_type(message, output)]
    return tf.group(*summary_op)


def add_check_numerics_ops():
    """Connect a `check_numerics` to every floating point tensor.
    `check_numerics` operations themselves are added for each `half`, `float`,
    or `double` tensor in the graph. For all ops in the graph, the
    `check_numerics` op for all of its (`half`, `float`, or `double`) inputs
    is guaranteed to run before the `check_numerics` op on any of its outputs.
    Returns:
      A `group` op depending on all `check_numerics` ops added.

    Based on `tf.add_check_numerics_ops`; modified to work around problem with
    variables in different "frames" (triggered by attempt to merge nodes
    from inside and outside the while loop of an RNN).
    """
    check_op = []
    # This code relies on the ordering of ops in get_operations().
    # The producer of a tensor always comes before that tensor's consumer in
    # this list. This is true because get_operations() returns ops in the order
    # added, and an op can only be added after its inputs are added.
    for op in tf.get_default_graph().get_operations():
        if op.name and any(re.search(pattern, op.name) for pattern in NO_MONITOR):
            continue
        for output in op.outputs:
            if output.dtype in [tf.float16, tf.float32, tf.float64] and \
                    output.op._get_control_flow_context() == \
                    tf.get_default_graph()._get_control_flow_context():
                message = op.name + ":" + str(output.value_index)
                with tf.control_dependencies(check_op):
                    check_op = [tf.check_numerics(output, message=message)]
    return tf.group(*check_op)


def print_shape(t):
    print_node = tf.Print(t, [tf.shape(t)], message='{}: '.format(t))
    with tf.control_dependencies([print_node]):
        attached = tf.identity(t)
    return attached


def check_numerics(t):
    check_op = tf.check_numerics(t, message='{}: '.format(t))
    with tf.control_dependencies([check_op]):
        attached = tf.identity(t)
    return attached


def dynamic_rnn_decoder(cell, decoder_fn, inputs=None, sequence_lengths=None,
                        parallel_iterations=None, swap_memory=False,
                        time_major=False, scope=None, name=None):
    """(Stolen from an old version of Tensorflow r1.0 [2/24/2017]. Replace with
    the official API when we feel comfortable switching everything to r1.0.)

    Dynamic RNN decoder for a sequence-to-sequence model specified by
    RNNCell and decoder function.
    The `dynamic_rnn_decoder` is similar to the `tf.python.ops.rnn.dynamic_rnn`
    as the decoder does not make any assumptions of sequence length and batch
    size of the input.
    The `dynamic_rnn_decoder` has two modes: training or inference and expects
    the user to create seperate functions for each.
    Under both training and inference `cell` and `decoder_fn` is expected. Where
    the `cell` performs computation at every timestep using the `raw_rnn` and
    the `decoder_fn` allows modelling of early stopping, output, state, and next
    input and context.
    When training the user is expected to supply `inputs`. At every time step a
    slice of the supplied input is fed to the `decoder_fn`, which modifies and
    returns the input for the next time step.
    `sequence_lengths` is optional an only used if the `decoder_fn` returns
    `None` for early stopping. `sequence_lengths` determines how many time steps
    to compute. `sequence_lengths` allows variable length samples in a batch by
    early stopping and is usually used when training. If `inputs` is not `None`
    and sequence_lengths=None` it is inferred from the `inputs` as the maximal
    possible sequence length.
    Under inference `inputs` is expected to be `None` and the input is inferred
    solely from the `decoder_fn`.
    Args:
        cell: An instance of RNNCell.
        decoder_fn: A function that takes time, cell state, cell input,
            cell output and context state. It returns a early stopping vector,
            cell state, next input, cell output and context state.
            Examples of decoder_fn can be found in the decoder_fn.py folder.
        inputs: The inputs for decoding (embedded format).
            If `time_major == False` (default), this must be a `Tensor` of shape:
                `[batch_size, max_time, ...]`.
            If `time_major == True`, this must be a `Tensor` of shape:
                `[max_time, batch_size, ...]`.
            The input to `cell` at each time step will be a `Tensor` with dimensions
                `[batch_size, ...]`.
        sequence_lengths: (optional) An int32/int64 vector sized `[batch_size]`.
        parallel_iterations: (Default: 32).  The number of iterations to run in
            parallel.  Those operations which do not have any temporal dependency
            and can be run in parallel, will be.  This parameter trades off
            time for space.  Values >> 1 use more memory but take less time,
            while smaller values use less memory but computations take longer.
        swap_memory: Transparently swap the tensors produced in forward inference
            but needed for back prop from GPU to CPU.  This allows training RNNs
            which would typically not fit on a single GPU, with very minimal (or no)
            performance penalty.
        time_major: The shape format of the `inputs` and `outputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        scope: VariableScope for the `raw_rnn`;
            defaults to None.
        name: NameScope for the decoder;
            defaults to "dynamic_rnn_decoder"
    Returns:
        A pair (outputs, state) where:
            outputs: the RNN output 'Tensor'.
                If time_major == False (default), this will be a `Tensor` shaped:
                    `[batch_size, max_time, cell.output_size]`.
                If time_major == True, this will be a `Tensor` shaped:
                    `[max_time, batch_size, cell.output_size]`.
            state: The final state and will be shaped
                         `[batch_size, cell.state_size]`.
    Raises:
        ValueError: if inputs is not None and has less than three dimensions.
        ValueError: if inputs is not None and has None in the last dimension.
    """
    with tf.name_scope(name, "dynamic_rnn_decoder",
                       [cell, decoder_fn, inputs, sequence_lengths,
                        parallel_iterations, swap_memory, time_major, scope]):
        if inputs is not None:
            # Convert to tensor
            inputs = tf.convert_to_tensor(inputs)

            # Test input dimensions
            if inputs.get_shape().ndims is not None and (
                    inputs.get_shape().ndims < 3):
                raise ValueError("Inputs must have at least three dimensions")
            if inputs.get_shape()[-1] is None:
                raise ValueError("Inputs must not be `None` in the feature (3'rd) "
                                 "dimension")
            # Setup of RNN (dimensions, sizes, length, initial state, dtype)
            if not time_major:
                # [batch, seq, features] -> [seq, batch, features]
                inputs = tf.transpose(inputs, perm=[1, 0, 2])

            dtype = inputs.dtype
            # Get data input information
            input_depth = int(inputs.get_shape()[2])
            batch_depth = inputs.get_shape()[1].value
            max_time = inputs.get_shape()[0].value
            if max_time is None:
                max_time = tf.shape(inputs)[0]
            # Setup decoder inputs as TensorArray
            inputs_ta = tf.TensorArray(dtype, size=max_time)
            inputs_ta = inputs_ta.unpack(inputs)

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_state is None:  # first call, before while loop (in raw_rnn)
                if cell_output is not None:
                    raise ValueError("Expected cell_output to be None when cell_state "
                                     "is None, but saw: %s" % cell_output)
                if loop_state is not None:
                    raise ValueError("Expected loop_state to be None when cell_state "
                                     "is None, but saw: %s" % loop_state)
                context_state = None
            else:  # subsequent calls, inside while loop, after cell excution
                if isinstance(loop_state, tuple):
                    (done, context_state) = loop_state
                else:
                    done = loop_state
                    context_state = None

            # call decoder function
            if inputs is not None:  # training
                # get next_cell_input
                if cell_state is None:
                    next_cell_input = inputs_ta.read(0)
                else:
                    if batch_depth is not None:
                        batch_size = batch_depth
                    else:
                        batch_size = tf.shape(done)[0]
                    next_cell_input = tf.cond(
                        tf.equal(time, max_time),
                        lambda: tf.zeros([batch_size, input_depth], dtype=dtype),
                        lambda: inputs_ta.read(time))
                (next_done, next_cell_state, next_cell_input, emit_output,
                 next_context_state) = decoder_fn(time, cell_state, next_cell_input,
                                                  cell_output, context_state)
            else:  # inference
                # next_cell_input is obtained through decoder_fn
                (next_done, next_cell_state, next_cell_input, emit_output,
                 next_context_state) = decoder_fn(time, cell_state, None, cell_output,
                                                  context_state)

            # check if we are done
            if next_done is None:  # training
                next_done = time >= sequence_lengths

            # build next_loop_state
            if next_context_state is None:
                next_loop_state = next_done
            else:
                next_loop_state = (next_done, next_context_state)

            return (next_done, next_cell_input, next_cell_state,
                    emit_output, next_loop_state)

        # Run raw_rnn function
        outputs_ta, state, _ = tf.nn.raw_rnn(
            cell, loop_fn, parallel_iterations=parallel_iterations,
            swap_memory=swap_memory, scope=scope
        )
        outputs = outputs_ta.pack()

        if not time_major:
            # [seq, batch, features] -> [batch, seq, features]
            outputs = tf.transpose(outputs, perm=[1, 0, 2])
        return outputs, state


def simple_decoder_fn_inference(output_fn, encoder_state, embeddings,
                                start_of_sequence_id, end_of_sequence_id,
                                maximum_length, num_decoder_symbols,
                                dtype=tf.int32, name=None):
    with tf.name_scope(name, "simple_decoder_fn_inference",
                       [output_fn, encoder_state, embeddings,
                        start_of_sequence_id, end_of_sequence_id,
                        maximum_length, num_decoder_symbols, dtype]):
        if not isinstance(encoder_state, tuple):
            encoder_state = tf.convert_to_tensor(encoder_state)
        start_of_sequence_id = tf.convert_to_tensor(start_of_sequence_id, dtype)
        end_of_sequence_id = tf.convert_to_tensor(end_of_sequence_id, dtype)
        maximum_length = tf.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = tf.convert_to_tensor(num_decoder_symbols, dtype)
        encoder_info = encoder_state[0] if isinstance(encoder_state, tuple) else encoder_state
        batch_size = encoder_info.get_shape()[0].value
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = tf.shape(encoder_info)[0]

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        with tf.name_scope(name, "simple_decoder_fn_inference",
                           [time, cell_state, cell_input, cell_output,
                            context_state]):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" %
                                 cell_input)
            if cell_output is None:
                # invariant that this is time == 0
                next_input_id = tf.ones([batch_size], dtype=dtype) * (
                    start_of_sequence_id)
                done = tf.zeros([batch_size], dtype=tf.bool)
                cell_state = encoder_state
                cell_output = tf.zeros([num_decoder_symbols],
                                       dtype=tf.float32)
            else:
                cell_output = output_fn(cell_output)
                next_input_id = tf.cast(
                    tf.argmax(cell_output, 1), dtype=dtype)
                done = tf.equal(next_input_id, end_of_sequence_id)
            next_input = tf.gather(embeddings, next_input_id)
            # if time > maxlen, return all true vector
            done = tf.cond(
                tf.greater(time, maximum_length),
                lambda: tf.ones([batch_size], dtype=tf.bool),
                lambda: done)
            return (done, cell_state, next_input, cell_output, context_state)
    return decoder_fn
