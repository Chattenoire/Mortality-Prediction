import tensorflow as tf
from copy import deepcopy


def copy_weights(w_list):
    """Deep-copy a list of NumPy arrays."""
    return [w.copy() for w in w_list]


class BaseAggregator:
    """
    ONE global sub-classed model is kept.
    For each client:
      • snapshot weights
      • train in-place
      • compute delta
      • restore snapshot
    """

    def __init__(self, model_fn, client_lr, server_lr=None, mu=None):
        self.global_model = model_fn()
        self.client_lr = client_lr
        self.server_lr = server_lr      # SCAFFOLD
        self.mu = mu                    # FedProx
        self.c = None                   # SCAFFOLD control variate

        self.global_model.compile(
            optimizer=tf.keras.optimizers.Adam(self.client_lr),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )

    # helper shared by all subclasses
    @staticmethod
    def _count_examples(dataset):
        return sum(1 for _ in dataset.unbatch())


# ───────────────────────────  FedAvg  ─────────────────────────── #
class FedAvgAggregator(BaseAggregator):

    def client_update(self, client_ds):
        orig = copy_weights(self.global_model.get_weights())

        for (xt, xs), y in client_ds:
            self.global_model.train_on_batch((xt, xs), y, reset_metrics=False)

        delta = [o - n for o, n in zip(orig, self.global_model.get_weights())]
        self.global_model.set_weights(orig)
        return delta, self._count_examples(client_ds)

    def server_update(self, pkgs):
        deltas, ns = zip(*pkgs)
        tot = float(sum(ns))
        mean_delta = [
            tf.add_n([d[i] * (n / tot) for d, n in zip(deltas, ns)])
            for i in range(len(deltas[0]))
        ]
        self.global_model.set_weights(
            [w - md for w, md in zip(self.global_model.get_weights(), mean_delta)]
        )


# ───────────────────────────  FedProx  ────────────────────────── #
class FedProxAggregator(BaseAggregator):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.opt = tf.keras.optimizers.Adam(self.client_lr)

        @tf.function
        def step(x_ts, x_sta, y, g_weights):
            with tf.GradientTape() as tape:
                logits = self.global_model((x_ts, x_sta), training=True)
                ce = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y, logits)
                prox = tf.add_n([
                    tf.reduce_sum(tf.square(w - gw))
                    for w, gw in zip(self.global_model.trainable_variables, g_weights)
                ])
                loss = ce + 0.5 * self.mu * prox
            grads = tape.gradient(loss, self.global_model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.global_model.trainable_variables))

        self._prox_step = step

    def client_update(self, client_ds):
        orig = copy_weights(self.global_model.get_weights())
        for (xt, xs), y in client_ds:
            self._prox_step(xt, xs, y, orig)

        delta = [o - n for o, n in zip(orig, self.global_model.get_weights())]
        self.global_model.set_weights(orig)
        return delta, self._count_examples(client_ds)

    server_update = FedAvgAggregator.server_update


# ───────────────────────────  SCAFFOLD  ───────────────────────── #
class ScaffoldAggregator(BaseAggregator):

    def __init__(self, model_fn, client_lr, server_lr):
        super().__init__(model_fn, client_lr, server_lr)
        self.c = [tf.zeros_like(v) for v in self.global_model.trainable_variables]
        self.opt = tf.keras.optimizers.Adam(self.client_lr)

        @tf.function
        def step(x_ts, x_sta, y, ci):
            with tf.GradientTape() as tape:
                logits = self.global_model((x_ts, x_sta), training=True)
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y, logits)
            grads = tape.gradient(loss, self.global_model.trainable_variables)
            corr = [g + cg - cig for g, cg, cig in zip(grads, self.c, ci)]
            self.opt.apply_gradients(zip(corr, self.global_model.trainable_variables))

        self._scaffold_step = step

    def client_update(self, client_ds):
        orig = copy_weights(self.global_model.get_weights())
        ci = [tf.zeros_like(v) for v in self.global_model.trainable_variables]

        for (xt, xs), y in client_ds:
            self._scaffold_step(xt, xs, y, ci)

        delta = [o - n for o, n in zip(orig, self.global_model.get_weights())]
        ci = [cij + delta_j / self.client_lr for cij, delta_j in zip(ci, delta)]

        self.global_model.set_weights(orig)
        return delta, ci, self._count_examples(client_ds)

    def server_update(self, pkgs):
        deltas, cis, ns = zip(*pkgs)
        tot = float(sum(ns))
        mean_delta = [
            tf.add_n([d[i] * (n / tot) for d, n in zip(deltas, ns)])
            for i in range(len(deltas[0]))
        ]
        self.global_model.set_weights(
            [w - md for w, md in zip(self.global_model.get_weights(), mean_delta)]
        )
        mean_ci = [tf.add_n(t) / len(cis) for t in zip(*cis)]
        self.c = [c + self.server_lr * (mc - c)
                  for c, mc in zip(self.c, mean_ci)]
