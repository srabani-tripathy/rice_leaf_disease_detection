from rldd.entity import TrainingConfig
import tensorflow as tf
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def train_valid_generator(self):

        data_dir = self.config.training_data
        seed = 123

        # Splitting data and creating training and validation datasets
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.1,
            subset="training",
            seed=seed,
            image_size=(self.config.params_image_size[0], self.config.params_image_size[1]),
            batch_size=self.config.params_batch_size
        )

        self.validation_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.1,
            subset="validation",
            seed=seed,
            image_size=(self.config.params_image_size[0], self.config.params_image_size[1]),
            batch_size=self.config.params_batch_size
        )

        # Normalizing Pixel Values for Training and Validation Datasets
        self.train_ds = self.train_ds.map(lambda x, y: (x / 255.0, y))
        self.validation_ds = self.validation_ds.map(lambda x, y: (x / 255.0, y))

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self, callback_list: list):

        self.model.fit(
            self.train_ds,
            epochs=self.config.params_epochs,
            validation_data=self.validation_ds,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )