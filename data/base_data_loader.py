import os
import tensorflow as tf
import tensorflow.keras.preprocessing as tfkp


class BaseDataLoader:
    def __init__(self, root_dir, validation_split, seed, train):
        self.root_dir = root_dir
        self.validation_split = validation_split if train else 0.0
        self.seed = seed
        self.train = train

    def load_dataset(self, dataset, image_size):
        full_dir = os.path.join(
            self.root_dir, dataset, "train" if self.train else "test"
        )

        try:
            lq_ds, gt_ds = tfkp.image_dataset_from_directory(
                full_dir,
                labels=None,
                color_mode="rgb",
                batch_size=None,
                image_size=image_size,
                shuffle=False,
                seed=self.seed,
                validation_split=0.5,
                subset="both",
                crop_to_aspect_ratio=True,
            )
        except:
            print(f"No Dataset found in {full_dir}.")
            pass

        return lq_ds, gt_ds
