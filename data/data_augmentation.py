import tensorflow as tf
import tensorflow.keras.layers as tfkl

AUTO = tf.data.AUTOTUNE

class PairedImageAugmentation(tfkl.Layer):
                def __init__(self):
                    super(PairedImageAugmentation, self).__init__()
                    # Define your augmentation parameters here
                    self.h_flip = tf.image.flip_left_right
                    self.v_flip = tf.image.flip_up_down
                    self.rotation = tf.image.rot90

                def call(self, inputs):
                    lq_img, gt_img = inputs
                    # Apply the same random flip and rotation to both images
                    h_flip_seed = np.random.randint(0, 2)
                    v_flip_seed = np.random.randint(0, 2)
                    rot_flip_seed = np.random.randint(0, 4)
                    if h_flip_seed == 1:
                        lq_img = self.h_flip(lq_img)
                        gt_img = self.h_flip(gt_img)
                    if v_flip_seed == 1:
                        lq_img = self.v_flip(lq_img)
                        gt_img = self.v_flip(gt_img)
                    if rot_flip_seed > 0:
                        for i in range(rot_flip_seed+1):
                            lq_img = self.rotation(lq_img)
                            gt_img = self.rotation(gt_img)
                    return lq_img, gt_img


def apply_augmentation(dataset):
    custom_augmentation = PairedImageAugmentation()

    def data_augmentation(lq_img, gt_img):
        normalized_lq = tfkl.Rescaling(1.0 / 255)(lq_img)
        normalized_gt = tfkl.Rescaling(1.0 / 255)(gt_img)
        # Apply custom augmentation
        augmented_lq, augmented_gt = custom_augmentation((normalized_lq, normalized_gt))
        return augmented_lq, augmented_gt

    return dataset.map(lambda lq_img, gt_img: data_augmentation(lq_img, gt_img), num_parallel_calls=AUTO)
