import tensorflow as tf
from pathlib import Path
from rldd.entity import EvaluationConfig
from rldd.utils import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):
        
        data_dir = self.config.training_data
        seed = 123

        self.validation_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.1,
            subset="validation",
            seed=seed,
            image_size=(self.config.params_image_size[0], self.config.params_image_size[1]),
            batch_size=self.config.params_batch_size
        )

        self.validation_ds = self.validation_ds.map(lambda x, y: (x / 255.0, y))

    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self.validation_ds)

    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)