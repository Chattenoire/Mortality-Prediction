import tensorflow as tf
import numpy as np
from .rules import RuleBase

def make_masks(input_dim, k, R, seed):
    rng = np.random.default_rng(seed)
    masks = np.zeros((R, input_dim), dtype=np.float32)
    for i in range(R):
        idx = rng.choice(input_dim, size=k, replace=False)
        masks[i, idx] = 1.
    return masks

class FedFNN(tf.keras.Model):
    def __init__(self, ts_dim, static_dim, cfg):
        super().__init__()
        p = cfg['model']
        self.gru1 = tf.keras.layers.GRU(p['gru_units'], return_sequences=True)
        self.gru2 = tf.keras.layers.GRU(p['gru_units'], return_sequences=True)
        self.att = tf.keras.layers.Attention()
        self.drop = tf.keras.layers.Dropout(p['dropout'])

        fused_dim = p['gru_units'] + static_dim
        masks = make_masks(fused_dim,
                           cfg['run']['k'],
                           cfg['run']['R'],
                           cfg['seed'])
        self.rules = [RuleBase(fused_dim, m, p['lambda_val'], name=f"rule_{i}")
                      for i, m in enumerate(masks)]
        self.norm = tf.keras.layers.LayerNormalization()
        self.final = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        ts, stat = inputs
        h = self.gru1(ts)
        h = self.gru2(h)
        h = self.att([h, h])
        h = tf.reduce_mean(h, axis=1)          # simple mean for demo
        h = self.drop(h, training=training)

        x = tf.concat([h, stat], axis=1)
        x = self.norm(x)

        fires, outs = zip(*(r(x) for r in self.rules))
        fires = tf.concat(fires, axis=1)       # (B,R)
        outs  = tf.concat(outs,  axis=1)       # (B,R)

        fires_sum = tf.reduce_sum(fires, axis=1, keepdims=True) + 1e-9
        norm_fires = fires / fires_sum
        fuzzy_out = tf.reduce_sum(norm_fires * outs, axis=1, keepdims=True)
        return self.final(fuzzy_out)
