import os
import numpy as np
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.utils as tfku

from .base_data_loader import BaseDataLoader


class FujiSonyDataLoader(BaseDataLoader):
    def load_dataset(self, dataset, image_size):
        lq_ds, lq_val_ds, _, _ = super().load_dataset(dataset, image_size)
        full_dir = os.path.join(
            self.root_dir, dataset, "train" if self.train else "test"
        )

        all_images = []
        for img_path in tqdm(lq_ds.file_paths + lq_val_ds.file_paths):
            img_id = os.path.basename(img_path).split("_")[0]
            try:
                img = tfku.load_img(
                    os.path.join(full_dir, "target", f"{img_id}_00_10s.png"),
                    target_size=image_size,
                )
            except FileNotFoundError:
                img = tfku.load_img(
                    os.path.join(full_dir, "target", f"{img_id}_00_30s.png"),
                    target_size=image_size,
                )

            img_array = tfku.img_to_array(img)
            all_images.append(img_array)

        all_images_np = np.array(all_images)
        num_val_samples = int(self.validation_split * len(all_images_np))
        gt_ds = tf.data.Dataset.from_tensor_slices(all_images_np[:-num_val_samples])
        gt_val_ds = tf.data.Dataset.from_tensor_slices(all_images_np[-num_val_samples:])

        return lq_ds, lq_val_ds, gt_ds, gt_val_ds


class RellisurDataloader(BaseDataLoader):
    def load_dataset(self, dataset, image_size):
        lq_ds, lq_val_ds, _, _ = super().load_dataset(dataset, image_size)
        full_dir = os.path.join(
            self.root_dir, dataset, "train" if self.train else "test"
        )

        all_images = []
        for img_path in tqdm(lq_ds.file_paths + lq_val_ds.file_paths):
            img_id = os.path.basename(img_path).split("-")[0]
            img = tfku.load_img(
                os.path.join(full_dir, "target", f"{img_id}.png"),
                target_size=image_size,
            )
            img_array = tfku.img_to_array(img)
            all_images.append(img_array)

        all_images_np = np.array(all_images)
        num_val_samples = int(self.validation_split * len(all_images_np))
        gt_ds = tf.data.Dataset.from_tensor_slices(all_images_np[:-num_val_samples])
        gt_val_ds = tf.data.Dataset.from_tensor_slices(all_images_np[-num_val_samples:])

        return lq_ds, lq_val_ds, gt_ds, gt_val_ds


class SiceDataLoader(BaseDataLoader):
    def load_dataset(self, dataset, image_size):
        lq_ds, lq_val_ds, _, _ = super().load_dataset(dataset, image_size)
        full_dir = os.path.join(
            self.root_dir, dataset, "train" if self.train else "test"
        )

        all_images = []
        for img_path in tqdm(lq_ds.file_paths + lq_val_ds.file_paths):
            img_id = os.path.basename(img_path).split("_")[0]
            img = tfku.load_img(
                os.path.join(full_dir, "target", f"{img_id}_gt.JPG"),
                target_size=image_size,
            )
            img_array = tfku.img_to_array(img)
            all_images.append(img_array)

        all_images_np = np.array(all_images)
        num_val_samples = int(self.validation_split * len(all_images_np))
        gt_ds = tf.data.Dataset.from_tensor_slices(all_images_np[:-num_val_samples])
        gt_val_ds = tf.data.Dataset.from_tensor_slices(all_images_np[-num_val_samples:])

        return lq_ds, lq_val_ds, gt_ds, gt_val_ds
