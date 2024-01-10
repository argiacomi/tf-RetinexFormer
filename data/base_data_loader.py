import os
import tensorflow as tf
import tensorflow.keras.preprocessing as tfkp


class BaseDataLoader:
    def __init__(self, root_dir, validation_split, seed, train=True):
        self.root_dir = root_dir
        self.validation_split = validation_split if train else 0.0
        self.seed = seed
        self.train = train

    def load_dataset(self, dataset, image_size):
        full_dir = os.path.join(
            self.root_dir, dataset, "train" if self.train else "test"
        )

        lq_ds = tfkp.image_dataset_from_directory(
            os.path.join(full_dir, "input"),
            labels=None,
            color_mode="rgb",
            batch_size=None,
            image_size=image_size,
            shuffle=False,
            seed=self.seed,
            validation_split=self.validation_split,
            subset="training",
        )

        lq_val_ds = tfkp.image_dataset_from_directory(
            os.path.join(full_dir, "input"),
            labels=None,
            color_mode="rgb",
            batch_size=None,
            image_size=image_size,
            shuffle=False,
            seed=self.seed,
            validation_split=self.validation_split,
            subset="validation",
        )

        gt_ds = tfkp.image_dataset_from_directory(
            os.path.join(full_dir, "target"),
            labels=None,
            color_mode="rgb",
            batch_size=None,
            image_size=image_size,
            shuffle=False,
            seed=self.seed,
            validation_split=self.validation_split,
            subset="training",
        )

        gt_val_ds = tfkp.image_dataset_from_directory(
            os.path.join(full_dir, "target"),
            labels=None,
            color_mode="rgb",
            batch_size=None,
            image_size=image_size,
            shuffle=False,
            seed=self.seed,
            validation_split=self.validation_split,
            subset="validation",
        )

        return lq_ds, lq_val_ds, gt_ds, gt_val_ds
