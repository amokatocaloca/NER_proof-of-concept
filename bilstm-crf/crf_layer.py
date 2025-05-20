# crf_layer.py
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K
import keras

@tf.keras.utils.register_keras_serializable(package="Custom", name="CRF")
class CRF(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        _, self.seq_len, self.num_tags = input_shape
        self.transition = self.add_weight(
            name="transitions",
            shape=(self.num_tags, self.num_tags),
            initializer="zeros"
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        # mask emissions
        if mask is not None:
            return inputs * tf.cast(mask[..., None], tf.float32)
        return inputs

    def compute_output_shape(self, input_shape):
        # we still output [B,T,K] to feed into loss
        return input_shape

    def compute_loss(self, y_true, emissions):
        mask   = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        y_true = tf.cast(y_true, tf.int32)
        ll     = self._compute_log_likelihood(y_true, emissions, mask)
        return -tf.reduce_mean(ll)
    
    def _compute_log_likelihood(self, y_true, emissions, mask):
        B = tf.shape(emissions)[0]
        T = tf.shape(emissions)[1]

        # 1) emission scores for gold path
        em_masked = emissions * mask[:, :, None]
        idxs      = self._get_indices(y_true, B, T)
        e_scores  = tf.reduce_sum(tf.gather_nd(em_masked, idxs), axis=1)

        # 2) transition scores
        t_scores = self._compute_transition_scores(y_true, mask)

        # 3) partition function
        log_z    = self._compute_partition(emissions, mask)
        return (e_scores + t_scores) - log_z

    def _compute_transition_scores(self, y_true, mask):
        prev = y_true[:, :-1]
        nxt  = y_true[:, 1:]
        m    = mask[:, 1:]
        idx  = tf.stack([prev, nxt], axis=-1)
        s    = tf.gather_nd(self.transition, idx)
        return tf.reduce_sum(s * m, axis=1)

    def _compute_partition(self, emissions, mask):
        # use your existing while-loop code here; unchanged
        seq_len = tf.shape(emissions)[1]
        alpha   = tf.TensorArray(tf.float32, size=seq_len)
        alpha   = alpha.write(0, emissions[:, 0, :])
        def step(t, ta):
            prev   = ta.read(t-1)
            emit_t = emissions[:, t, :]
            score  = prev[:,:,None] + self.transition + emit_t[:,None,:]
            new_a  = tf.reduce_logsumexp(score, axis=1)
            m_t    = mask[:, t][:,None]
            new_a  = new_a * m_t + prev*(1-m_t)
            return t+1, ta.write(t, new_a)

        _, alpha = tf.while_loop(
            lambda t, *_: t < seq_len,
            step,
            loop_vars=(1, alpha),
            maximum_iterations=seq_len-1
        )
        final = alpha.read(seq_len-1)
        return tf.reduce_logsumexp(final, axis=1)
    

    def _get_indices(self, y_true, B, T):
        b = tf.tile(tf.range(B)[:,None], [1,T])
        t = tf.tile(tf.range(T)[None,:], [B,1])
        return tf.stack([b, t, y_true], axis=-1)

    def get_config(self):
        return {}



@tf.keras.utils.register_keras_serializable(package="Custom", name="masked_accuracy")
def masked_accuracy(y_true, y_pred):
    y_true      = tf.cast(y_true, tf.int32)
    y_pred_tags = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    mask        = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    matches     = tf.cast(tf.equal(y_true, y_pred_tags), tf.float32) * mask
    return tf.reduce_sum(matches) / tf.reduce_sum(mask)


@keras.saving.register_keras_serializable(package="Custom", name="MaskedF1Score")
class MaskedF1Score(Metric):
    def __init__(self, num_classes, name='masked_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes     = num_classes
        self.true_positives  = self.add_weight(name='tp', shape=(), dtype='float32', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(), dtype='float32', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(), dtype='float32', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_tags = tf.argmax(y_pred, axis=-1)
        y_pred_oh   = tf.one_hot(y_pred_tags, self.num_classes)
        y_true_oh   = tf.one_hot(y_true,     self.num_classes)
        mask        = tf.cast(tf.not_equal(y_true, 0), tf.float32)[...,None]

        self.true_positives.assign_add(tf.reduce_sum(y_pred_oh * y_true_oh * mask))
        self.false_positives.assign_add(tf.reduce_sum(y_pred_oh * (1 - y_true_oh) * mask))
        self.false_negatives.assign_add(tf.reduce_sum((1 - y_pred_oh) * y_true_oh * mask))

    def result(self):
        p = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        r = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        return 2 * p * r / (p + r + K.epsilon())

    def reset_states(self):
        K.set_value(self.true_positives,  0)
        K.set_value(self.false_positives, 0)
        K.set_value(self.false_negatives, 0)



def viterbi_decode_batch(emissions, transition, mask):
    """
    emissions: np.array shape [batch, seq_len, K]
    transition: np.array shape [K, K]
    mask:       np.array shape [batch, seq_len]  (bool or 0/1)
    returns:    np.array shape [batch, seq_len] of best tag indices
    """
    import numpy as np
    B, T, K = emissions.shape
    paths = np.zeros((B, T), dtype=np.int32)

    for b in range(B):
        # dynamic programming for a single sequence
        dp = emissions[b,0]  # [K]
        backp = np.zeros((T, K), dtype=np.int32)
        for t in range(1, T):
            if not mask[b,t]:
                # pad – just carry forward best from previous
                backp[t] = 0
                continue
            scores = dp[:, None] + transition + emissions[b,t][None, :]
            best_i = np.argmax(scores, axis=0)   
            dp     = np.max(scores, axis=0)        
            backp[t] = best_i

        # backtrace
        last = int(np.argmax(dp))
        path = [last]
        for t in range(T-1, 0, -1):
            last = backp[t, last]
            path.append(last)
        paths[b] = path[::-1]
    return paths