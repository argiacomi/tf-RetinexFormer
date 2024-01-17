import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

AUTO = tf.data.AUTOTUNE


class PairedImageAugmentation(tf.keras.layers.Layer):
    def __init__(self):
        super(PairedImageAugmentation, self).__init__()
        self.h_flip = tf.image.flip_left_right
        self.v_flip = tf.image.flip_up_down

    def call(self, inputs, seed=None):
        lq_img_batch, gt_img_batch = inputs

        # Use TensorFlow's random functions to generate a single seed for the whole batch
        h_flip_seed = tf.random.uniform(
            [], seed=seed, minval=0, maxval=2, dtype=tf.int32
        )
        v_flip_seed = tf.random.uniform(
            [], seed=seed, minval=0, maxval=2, dtype=tf.int32
        )
        rot_flip_seed = tf.random.uniform(
            [], seed=seed, minval=0, maxval=4, dtype=tf.int32
        )

        # Apply the same random flip and rotation to the whole batch
        lq_img_batch = tf.cond(
            h_flip_seed == 1, lambda: self.h_flip(lq_img_batch), lambda: lq_img_batch
        )
        gt_img_batch = tf.cond(
            h_flip_seed == 1, lambda: self.h_flip(gt_img_batch), lambda: gt_img_batch
        )

        lq_img_batch = tf.cond(
            v_flip_seed == 1, lambda: self.v_flip(lq_img_batch), lambda: lq_img_batch
        )
        gt_img_batch = tf.cond(
            v_flip_seed == 1, lambda: self.v_flip(gt_img_batch), lambda: gt_img_batch
        )

        lq_img_batch = tf.image.rot90(lq_img_batch, k=rot_flip_seed)
        gt_img_batch = tf.image.rot90(gt_img_batch, k=rot_flip_seed)

        return lq_img_batch, gt_img_batch


def apply_augmentation(dataset, seed=None):
    custom_augmentation = PairedImageAugmentation()

    def data_augmentation(lq_img_batch, gt_img_batch, seed=seed):
        # Normalize the batch
        normalized_lq_batch = lq_img_batch / 255.0
        normalized_gt_batch = gt_img_batch / 255.0

        # Apply custom augmentation to the batch
        augmented_lq_batch, augmented_gt_batch = custom_augmentation(
            (normalized_lq_batch, normalized_gt_batch), seed=seed
        )
        return augmented_lq_batch, augmented_gt_batch

    return dataset.map(
        lambda lq_img_batch, gt_img_batch: data_augmentation(
            lq_img_batch, gt_img_batch, seed=seed
        ),
        num_parallel_calls=AUTO,
    )
