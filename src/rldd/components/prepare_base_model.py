from pathlib import Path
import tensorflow as tf
from rldd.entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.applications.DenseNet121(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            pooling=self.config.params_pooling,
            include_top=self.config.params_include_top
        )

        self.model.trainable = False

        self.save_model(path=self.config.base_model_path, model=self.model)

    
    
    @staticmethod
    def _prepare_full_model(model, classes, learning_rate):

        full_model = tf.keras.models.Sequential()
        full_model.add(model)

        full_model.add(tf.keras.layers.BatchNormalization())
        full_model.add(tf.keras.layers.Dropout(0.35))
        full_model.add(tf.keras.layers.Dense(220, activation="relu"))
        full_model.add(tf.keras.layers.Dense(classes, activation="softmax"))

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        full_model.summary()
        return full_model



    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
