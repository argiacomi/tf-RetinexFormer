import math

import tensorflow as tf
import tensorflow.keras.optimizers as tfko


class CosineDecayCycleRestarts(tfko.schedules.LearningRateSchedule):
    def __init__(
        self,
        base_lr,
        first_decay_steps,
        t_mul=2.0,
        m_mul=1.0,
        alpha=[0.0, 0.0],
        name=None,
    ):
        super().__init__()
        self.base_lr = base_lr
        self.first_decay_steps = first_decay_steps
        self._t_mul = t_mul
        self._m_mul = m_mul
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "SGDRDecay") as name:
            base_lr = tf.convert_to_tensor(self.base_lr, name="base_lr")
            dtype = base_lr.dtype
            first_decay_steps = tf.cast(self.first_decay_steps, dtype)
            t_mul = tf.cast(self._t_mul, dtype)
            m_mul = tf.cast(self._m_mul, dtype)

            global_step_recomp = tf.cast(step, dtype)
            completed_fraction = global_step_recomp / first_decay_steps

            alpha = tf.cond(
                tf.greater(completed_fraction, 1.0),
                lambda: tf.cast(self.alpha[1], dtype),
                lambda: tf.cast(self.alpha[0], dtype),
            )

            def compute_step(completed_fraction, geometric=False):
                """Helper for `cond` operation."""
                if geometric:
                    i_restart = tf.floor(
                        tf.math.log(1.0 - completed_fraction * (1.0 - t_mul))
                        / tf.math.log(t_mul)
                    )

                    sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
                    completed_fraction = (
                        completed_fraction - sum_r
                    ) / t_mul**i_restart

                else:
                    i_restart = tf.floor(completed_fraction)
                    completed_fraction -= i_restart

                return i_restart, completed_fraction

            i_restart, completed_fraction = tf.cond(
                tf.equal(t_mul, 1.0),
                lambda: compute_step(completed_fraction, geometric=False),
                lambda: compute_step(completed_fraction, geometric=True),
            )

            m_fac = m_mul**i_restart
            cosine_decayed = (
                0.5
                * m_fac
                * (1.0 + tf.cos(tf.constant(math.pi, dtype=dtype) * completed_fraction))
            )
            decayed = (1 - alpha) * cosine_decayed + alpha

            return tf.multiply(base_lr, decayed, name=name)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "name": self.name,
        }
