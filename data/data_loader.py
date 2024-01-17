import os
import PIL.Image as Image
import numpy as np
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
    root_dir,
    dataset,
    image_size,
    validation_split=0.2,
    seed=None,
    shuffle=True,
    train=True,
):
    # if dataset in ["Fuji", "Sony"]:
    #     loader = FujiSonyDataLoader(root_dir, validation_split, seed, train)
    # elif dataset in ["RELLISUR"]:
    #     loader = RellisurDataloader(root_dir, validation_split, seed, train)
    # elif dataset in ["SICE"]:
    #     loader = SiceDataLoader(root_dir, validation_split, seed, train)
    # else:
    loader = BaseDataLoader(root_dir, validation_split, seed, train)

    # Load a regular dataset
    lq_ds, gt_ds = loader.load_dataset(dataset, image_size)

    lq_fp = lq_ds.file_paths
    gt_fp = gt_ds.file_paths

    assert lq_fp == [
        path.replace("target", "input").replace("-gt", "") for path in gt_fp
    ], f"Mismatch in dataset alignment for {dataset}: lq {lq_fp} != gt {gt_fp}"

    train_ds = tf.data.Dataset.zip(lq_ds, gt_ds)
    if train:
        buffer_div = lq_ds.element_spec.shape[1] / 640

        data_size = train_ds.cardinality()
        train_size = round(data_size.numpy() * (1 - validation_split))
        buffer_size = min(data_size.numpy(), data_size.numpy() // buffer_div)

        if shuffle:
            print(f"Shuffling: {dataset} {str(image_size)} dataset")
            train_ds = train_ds.shuffle(
                buffer_size=buffer_size,
                seed=seed,
                reshuffle_each_iteration=True,
            )

        val_ds = train_ds.skip(train_size)
        train_ds = train_ds.take(train_size)

        return train_ds, val_ds
    else:
        return train_ds, None


def prepare_train_dataset(
    root_dir,
    save_dir,
    save_ds,
    data_dirs,
    batch_size,
    validation_split=0.2,
    seed=None,
    shuffle=True,
    augment=True,
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
                temp_train_ds, temp_val_ds = load_datasets(
                    root_dir,
                    dataset,
                    image_size,
                    validation_split=validation_split,
                    seed=seed,
                    shuffle=shuffle,
                    train=True,
                )
                print(f"Batching: {dataset} {str(image_size)} dataset")
                temp_train_ds = temp_train_ds.batch(batch_size["train"])
                temp_val_ds = temp_val_ds.batch(batch_size["val"])

                if augment:
                    print(f"Augmenting: {dataset} {str(image_size)} dataset")
                    temp_train_ds = apply_augmentation(temp_train_ds, seed=seed)

                if save_ds:
                    print(f"Saving: {dataset} {str(image_size)} dataset")
                    temp_train_ds.save(dataset_train_path, compression="GZIP")
                    temp_val_ds.save(dataset_val_path, compression="GZIP")

            if "train_ds" in locals():
                train_ds = train_ds.concatenate(temp_train_ds)
            else:
                print(f"Concatenating dataset: {dataset} {str(image_size)}")
                train_ds = temp_train_ds

            if "val_ds" in locals():
                val_ds = val_ds.concatenate(temp_val_ds)
            else:
                val_ds = temp_val_ds
        # if save_ds:
        #     train_ds.save(os.path.join(save_dir, "train/ALL"), compression="GZIP")
        #     val_ds.save(os.path.join(save_dir, "val/ALL"), compression="GZIP")
    return train_ds, val_ds


def prepare_test_dataset(root_dir, save_dir, save_ds, data_dirs, batch_size, seed=None):
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
                temp_test_ds, _ = load_datasets(
                    root_dir,
                    dataset,
                    image_size,
                    validation_split=0,
                    seed=seed,
                    train=False,
                )
                print(f"Batching: {dataset} {str(image_size)} dataset")
                temp_test_ds = temp_test_ds.batch(batch_size["val"])

                if save_ds:
                    print(f"Saving: {dataset} {str(image_size)} dataset")
                    temp_test_ds.save(dataset_test_path, compression="GZIP")

            if "test_ds" in locals():
                print(f"Concatenating dataset: {dataset} {str(image_size)}")
                test_ds = test_ds.concatenate(temp_test_ds)
            else:
                test_ds = temp_test_ds
        if save_ds:
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
        print("Instantiating DataLoader...")
        self.root_dir = options["root_dir"]
        self.save_dir = options["save_dir"]
        self.data_dirs = options["data_dirs"]
        self.validation_split = options["validation_split"]
        self.batch_size = options["batch_size"]
        self.use_augment = options["use_augment"]
        self.use_shuffle = options["use_shuffle"]
        self.save_ds = options["save_ds"]
        self.seed = seed

    def load_train_data(self):
        train_ds, val_ds = prepare_train_dataset(
            self.root_dir,
            self.save_dir,
            self.save_ds,
            self.data_dirs,
            self.batch_size,
            self.validation_split,
            self.seed,
            self.use_shuffle,
            self.use_augment,
        )

        return (
            train_ds.prefetch(tf.data.AUTOTUNE),
            val_ds.prefetch(tf.data.AUTOTUNE),
        )

    def load_test_data(self):
        test_ds = prepare_test_dataset(
            self.root_dir,
            self.save_dir,
            self.save_ds,
            self.data_dirs,
            self.batch_size,
            self.seed,
        )

        return test_ds.cache().prefetch(tf.data.AUTOTUNE)

    def load_predict_data(self, predict_dir):
        return prepare_predict_dataset(predict_dir)
