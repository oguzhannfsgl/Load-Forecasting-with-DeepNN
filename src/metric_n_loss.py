# Losses and metrics used to validate the performance of the model
# Since some values becomes zero (and some close to zero) and some metrics (mape) have
# this values on denominator, unstable metrics values are seen. Thus, unnormalized metrics
# are used.

import tensorflow as tf
import numpy as np



def unnormalize(x, load_min, load_max):
    return load_min + x * (load_max - load_min)
    
def get_unnormalized_rmse(load_min, load_max):
    def rmse(y_true, y_pred):
        y_true = unnormalize(y_true, load_min, load_max)
        y_pred = unnormalize(y_pred, load_min, load_max)
        return tf.math.sqrt(tf.reduce_mean((y_true - y_pred)**2))
    return rmse
    

def mape(y_true, y_pred):
    return tf.reduce_mean(tf.math.abs((y_pred - y_true)) / y_true * 100)


def get_unnormalized_mape(load_min, load_max):
    def unnormalized_mape(y_true, y_pred):
        y_true = unnormalize(y_true, load_min, load_max)
        y_pred = unnormalize(y_pred, load_min, load_max)
        return tf.reduce_mean(tf.math.abs((y_pred - y_true)) / y_true * 100)
    return unnormalized_mape


def mae(y_true, y_pred):
    y_true = y_true[0]
    loss = tf.math.reduce_mean(tf.math.abs(y_true - y_pred))
    return loss