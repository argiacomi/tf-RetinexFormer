import os

import tensorflow as tf
import tensorflow.keras.callbacks as tfkc
import tensorflow.keras.optimizers as tfko
import tensorflow.keras.utils as tfku

from model.RetinexFormerArch import RetinexFormer
from model.scheduler import CosineDecayCycleRestarts
from data import DataLoader
from metrics import PSNR

# tf.config.optimizer.set_experimental_options({
# "auto_mixed_precision_cpu": 1,
# "auto_parallel": 1,
# "auto_mixed_precision_mkl": 1,
# "dependency_optimization": 1,
# "remapping": 0,
# "layout_optimizer": 0,
# "pin_to_host_optimization": 1,
# "constant_folding": 1,
# "debug_stripper": 1,
# "scoped_allocator_optimization": 1,
# "loop_optimization": 1,
# "memory_optimization": 1,
# "auto_mixed_precision_onednn_bfloat16": 1,
# "arithmetic_optimization": 1,
# "common_subgraph_elimination": 1,
# "implementation_selector": 1,
# "function_optimization": 1,
# "shape_optimization": 1,
# })


class PrintLR(tfkc.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        print(f"Learning rate for epoch {epoch + 1} is {lr:.6f}")


class RetinexFormerModel:
    def __init__(self, options):
        self.options = options
        self.seed = self.options["manual_seed"]
        self.checkpoint_dir = (
            self.options["checkpoint_dir"]
            if "checkpoint_dir" in self.options
            else "./model/training_checkpoints"
        )
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch:04d}")
        self.initial_epoch = 0
        self.logs_dir = self.options["logs_dir"]
        self.dataset_options = self.options["dataset"]
        self.model_options = self.options["model"]
        self.training_options = self.options["training"]
        self.data_loader = DataLoader(self.dataset_options, self.seed)
        self.model = self.create_model()

    def create_model(self):
        tfku.set_random_seed(self.seed)
        model = RetinexFormer(**self.model_options)
        return model

    def compile_model(self):
        first_decay_steps = (
            self.training_options["scheduler"]["periods"][0]
            // self.dataset_options["batch_size"]
        )
        base_lr = self.training_options["optimizer"]["lr"]
        t_mul = (
            self.training_options["scheduler"]["periods"][1]
            / self.training_options["scheduler"]["periods"][0]
        )
        m_mul = self.training_options["scheduler"]["m_mul"]
        alpha = [
            alpha / base_lr for alpha in self.training_options["scheduler"]["lr_mins"]
        ]
        clipnorm = 0.01 if self.training_options["optimizer"]["clipnorm"] else None
        beta_1 = self.training_options["optimizer"]["betas"][0]
        beta_2 = self.training_options["optimizer"]["betas"][1]

        learning_rate_schedule = CosineDecayCycleRestarts(
            first_decay_steps=first_decay_steps,
            base_lr=base_lr,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
        )
        self.model.compile(
            optimizer=tfko.Adam(
                learning_rate=learning_rate_schedule,
                beta_1=beta_1,
                beta_2=beta_2,
                global_clipnorm=clipnorm,
            ),
            loss="mae",
            metrics=["accuracy", PSNR()],
        )

    def load_weights(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)

        if latest_checkpoint:
            print("Loading weights from ", latest_checkpoint)
            self.model.load_weights(latest_checkpoint)
            checkpoint_name = os.path.basename(latest_checkpoint)
            self.initial_epoch = int(checkpoint_name.split("_")[-1])

    def train(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        train_ds, val_ds = self.data_loader.load_train_data()
        epochs = max(
            1,
            self.training_options["total_iter"]
            // (train_ds.cardinality().numpy() * self.dataset_options["batch_size"]),
        )

        callbacks = [
            tfkc.ModelCheckpoint(
                filepath=self.checkpoint_prefix, verbose=1, save_weights_only=True
            ),
            tfkc.TensorBoard(
                log_dir=self.logs_dir, histogram_freq=1, profile_batch="500,520"
            ),
            tfkc.CSVLogger(os.path.join(self.logs_dir, "training.log")),
            PrintLR(),
        ]

        self.model.fit(
            train_ds,
            epochs=epochs,
            initial_epoch=self.initial_epoch,
            verbose="auto",
            validation_data=val_ds,
            shuffle=True,
            callbacks=callbacks,
        )

    def evaluate(self):
        test_ds = self.data_loader.load_test_data()

        callbacks = [
            tfkc.TensorBoard(log_dir=self.logs_dir, histogram_freq=1),
            tfkc.CSVLogger(os.path.join(self.logs_dir, "test.log")),
        ]

        return self.model.evaluate(
            test_ds,
            callbacks=callbacks,
        )

    def predict(self, data):
        predict_data = self.data_loader.load_predict_data(data)
        return self.model.predict(predict_data)
