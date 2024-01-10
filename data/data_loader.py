import os
from PIL import Image

import tensorflow as tf
import tensorflow.keras.preprocessing as tfkp

from .base_data_loader import BaseDataLoader
from .custom_data_loaders import (
    FujiSonyDataLoader,
    RellisurDataloader,
    SiceDataLoader,
)
from .data_augmentation import apply_augmentation


def load_datasets(
    root_dir, dataset, image_size, validation_split=0.2, seed=None, train=True
):
    if dataset in ["Fuji", "Sony"]:
        loader = FujiSonyDataLoader(root_dir, validation_split, seed, train)
    if dataset in ["RELLISUR"]:
        loader = RellisurDataloader(root_dir, validation_split, seed, train)
    if dataset in ["SICE"]:
        loader = SiceDataLoader(root_dir, validation_split, seed, train)
    else:
        loader = BaseDataLoader(root_dir, validation_split, seed, train)

    # Load a regular dataset
    lq_ds, lq_val_ds, gt_ds, gt_val_ds = loader.load_dataset(dataset, image_size)

    assert (
        lq_ds.cardinality() == gt_ds.cardinality()
    ), f"Mismatch in dataset lengths for {dir}: lq {lq_ds.cardinality()} != gt {gt_ds.cardinality()}"

    assert (
        lq_val_ds.cardinality() == gt_val_ds.cardinality()
    ), f"Mismatch in dataset lengths for {dir}: lq {lq_val_ds.cardinality()} != gt {gt_val_ds.cardinality()}"

    return lq_ds, gt_ds, lq_val_ds, gt_val_ds


def prepare_train_dataset(
    root_dir, save_dir, data_dirs, validation_split=0.2, seed=None, augment=True
):
    train_save_dir = os.path.join(save_dir, "train")
    val_save_dir = os.path.join(save_dir, "val")

    if os.path.exists(f"{train_save_dir}/ALL") and os.path.exists(
        f"{val_save_dir}/ALL"
    ):
        print(f"Loading cached datasets from {train_save_dir}/ALL")
        train_ds = tf.data.Dataset.load(f"{train_save_dir}/ALL", compression="GZIP")
        val_ds = tf.data.Dataset.load(f"{val_save_dir}/ALL", compression="GZIP")
    else:
        for dataset, image_size in data_dirs.items():
            print(f"Processing: {dataset} {str(image_size)} dataset")
            dataset_train_path = os.path.join(train_save_dir, dataset, str(image_size))
            dataset_val_path = os.path.join(val_save_dir, dataset, str(image_size))

            if os.path.exists(dataset_train_path) and os.path.exists(dataset_val_path):
                temp_train_ds = tf.data.Dataset.load(
                    dataset_train_path, compression="GZIP"
                )
                temp_val_ds = tf.data.Dataset.load(dataset_val_path, compression="GZIP")
            else:
                lq_ds, gt_ds, lq_val_ds, gt_val_ds = load_datasets(
                    root_dir,
                    dataset,
                    image_size,
                    validation_split=validation_split,
                    seed=seed,
                    train=True,
                )

                temp_train_ds = tf.data.Dataset.zip((lq_ds, gt_ds))
                temp_val_ds = tf.data.Dataset.zip((lq_val_ds, gt_val_ds))

                if augment:
                    temp_train_ds = apply_augmentation(temp_train_ds)

                temp_train_ds.save(dataset_train_path, compression="GZIP")
                temp_val_ds.save(dataset_val_path, compression="GZIP")

            if "train_ds" in locals():
                train_ds = train_ds.concatenate(temp_train_ds)
            else:
                train_ds = temp_train_ds

            if "val_ds" in locals():
                val_ds = val_ds.concatenate(temp_val_ds)
            else:
                val_ds = temp_val_ds
        train_ds.save(os.path.join(save_dir, "train/ALL"), compression="GZIP")
        val_ds.save(os.path.join(save_dir, "val/ALL"), compression="GZIP")
    return train_ds, val_ds


def prepare_test_dataset(root_dir, save_dir, data_dirs, seed=None):
    test_save_dir = os.path.join(save_dir, "test")
    if os.path.exists(f"{test_save_dir}/ALL"):
        print(f"Loading cached datasets from {test_save_dir}/ALL")
        test_ds = tf.data.Dataset.load(f"{test_save_dir}/ALL", compression="GZIP")
    else:
        for dataset, image_size in data_dirs.items():
            dataset_test_path = os.path.join(save_dir, "test", dataset, str(image_size))

            if os.path.exists(dataset_test_path):
                temp_test_ds = tf.data.Dataset.load(
                    dataset_test_path, compression="GZIP"
                )
            else:
                lq_ds, gt_ds, _, _ = load_datasets(
                    root_dir,
                    dataset,
                    image_size,
                    validation_split=0,
                    seed=seed,
                    train=False,
                )

                temp_test_ds = tf.data.Dataset.zip((lq_ds, gt_ds))
                temp_test_ds.save(dataset_test_path, compression="GZIP")

            if "test_ds" in locals():
                test_ds = test_ds.concatenate(temp_test_ds)
            else:
                test_ds = temp_test_ds
        test_ds.save(os.path.join(save_dir, "val/ALL"), compression="GZIP")

    return test_ds


def prepare_predict_dataset(directory):
    size = None

    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        if os.path.isfile(full_path):
            try:
                with Image.open(full_path) as img:
                    size = img.size
                    break
            except IOError:
                pass

    if size is None:
        raise ValueError("No valid images found in the directory.")

    dataset = tfkp.image_dataset_from_directory(
        directory,
        labels=None,
        color_mode="rgb",
        batch_size=None,
        image_size=size,
        shuffle=False,
    )

    return dataset


class DataLoader:
    def __init__(self, options, seed=None):
        self.root_dir = options["root_dir"]
        self.save_dir = options["save_dir"]
        self.data_dirs = options["data_dirs"]
        self.validation_split = options["validation_split"]
        self.batch_size = options["batch_size"]
        self.use_augment = options["use_augment"]
        self.use_shuffle = options["use_shuffle"]
        self.seed = seed

    def load_train_data(self):
        train_ds, val_ds = prepare_train_dataset(
            self.root_dir,
            self.save_dir,
            self.data_dirs,
            self.validation_split,
            self.seed,
            self.use_augment,
        )

        train_ds = train_ds.batch(self.batch_size).cache()
        val_ds = val_ds.batch(self.batch_size).cache()

        if self.use_shuffle:
            train_ds = train_ds.shuffle(
                buffer_size=train_ds.cardinality(), reshuffle_each_iteration=True
            )

        return (
            train_ds.prefetch(tf.data.AUTOTUNE),
            val_ds.cache().prefetch(tf.data.AUTOTUNE),
        )

    def load_test_data(self):
        test_ds = prepare_test_dataset(
            self.root_dir, self.save_dir, self.data_dirs, self.seed
        )
        test_ds = test_ds.batch(self.batch_size)
        return test_ds.cache().prefetch(tf.data.AUTOTUNE)

    def load_predict_data(self, predict_dir):
        return prepare_predict_dataset(predict_dir)
