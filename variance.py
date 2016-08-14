import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import tfutils


def moments(x):
    return tf.nn.moments(x, [0])[1]


'''
def naive(x):
    return tf.reduce_mean(x ** 2) - tf.reduce_mean(x) ** 2


def foil(x):
    rms = tf.sqrt(tf.reduce_mean(x ** 2))
    mean = tf.reduce_mean(x)
    return (rms - mean) * (rms + mean)


def shiftfirst(x):
    shifted = x - tf.reduce_mean(x)
    rms = tf.sqrt(tf.reduce_mean(shifted ** 2))
    mean = tf.reduce_mean(shifted)
    return (rms - mean) * (rms + mean)
'''


def tfu_moments(x):
    return tfutils.moments(x)[1]


def test_diff_one(v, x):
    arr_len = 4000
    inp = np.array([1.] + [0.] * (arr_len - 1))
    result = []
    max_diff = 0.0
    for shift in np.arange(1000.):
        var = v.eval(feed_dict={x: inp + shift})
        true = 1. / arr_len - 1. / arr_len ** 2
        diff = abs(var - true) / true
        max_diff = max(max_diff, diff)
        result.append(max_diff)
    return result


def test_diff_shift(v, x):
    arr_len = 4000
    inp = np.array([0.] + [1.] * (arr_len - 1))
    result = []
    max_diff = 0.0
    for shift in np.arange(1000.):
        var = v.eval(feed_dict={x: inp * shift})
        true = shift / arr_len * shift - (shift / arr_len) ** 2
        diff = abs(var - true) / true
        max_diff = max(max_diff, diff)
        result.append(max_diff)
    return result


def graph_numerical_stability():
    x = tf.placeholder(tf.float32, (None,))

    sess = tf.InteractiveSession()
    for test_func in [test_diff_one, test_diff_shift]:
        funcs = [moments, tfu_moments]  # [naive, foil, shiftfirst]
        with sess.as_default():
            for var_func in funcs:
                v = var_func(x)
                plt.semilogy(np.arange(1000), test_func(v, x))
        plt.legend([f.__name__ for f in funcs], loc='lower right')
        plt.show()


if __name__ == '__main__':
    graph_numerical_stability()
