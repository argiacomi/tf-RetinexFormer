import tensorflow as tf


class PSNR(tf.keras.metrics.Metric):
    def __init__(self, name="psnr", dtype=tf.float32, **kwargs):
        super(PSNR, self).__init__(name=name, dtype=dtype, **kwargs)
        self.psnr_sum = self.add_weight(name="psnr_sum", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)
        psnr = tf.cast(psnr, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, psnr.shape)
            psnr = tf.multiply(psnr, sample_weight)

        self.psnr_sum.assign_add(tf.reduce_sum(psnr))
        self.total_samples.assign_add(tf.cast(tf.size(psnr), self.dtype))

    def result(self):
        return self.psnr_sum / self.total_samples if self.total_samples != 0.0 else 0.0

    def reset_state(self):
        self.psnr_sum.assign(0)
        self.total_samples.assign(0)
