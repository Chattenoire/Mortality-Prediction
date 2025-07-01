import tensorflow as tf
import numpy as np
rng = np.random.default_rng()

def softmin(x, lam):
    w = tf.nn.softmax(-lam * x, axis=-1)
    return tf.reduce_sum(w * x, axis=-1, keepdims=True)

class RuleBase(tf.keras.layers.Layer):
    def __init__(self, input_dim, mask, lam=12.5, **kw):
        super().__init__(**kw)
        self.mask = tf.constant(mask.reshape(1, -1), tf.float32)
        self.lam = lam

        self.m     = tf.Variable(tf.zeros(input_dim), name="centres")
        # ------- NEW: sigma is always positive --------------------
        self._sigma_raw = tf.Variable(tf.ones(input_dim), name="sigma_raw")
        # ----------------------------------------------------------
        self.theta = self.add_weight(
            shape=(input_dim + 1, 1), initializer="glorot_uniform", name="consequent"
        )

    def call(self, x):
        masked_x = x * self.mask

        sigma = tf.nn.softplus(self._sigma_raw) + 1e-6   # > 0 for sure
        mu    = 1. - tf.abs(masked_x - self.m) / sigma
        mu    = tf.maximum(mu, 0.) * self.mask

        firing = softmin(mu, self.lam)

        x_aug  = tf.concat([tf.ones((tf.shape(x)[0], 1)), masked_x], axis=-1)
        out    = tf.matmul(x_aug, self.theta)
        return firing, out

