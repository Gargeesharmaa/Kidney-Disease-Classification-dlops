import os
import tensorflow as tf
from pathlib import Path
from Kidney_Disease_Classification.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        """Loads the prepared base model built in Stage 2."""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        """Prepares augmented training data streams and clean validation streams."""
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20 # 20% dedicated explicitly for validation
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Validation Generator ALWAYS stays clean (only rescaling)
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training Generator applies spatial distortions if AUGMENTATION is True
        if self.config.params_is_augmentation:
            print("Applying training data augmentation to prevent overfitting...")
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                fill_mode='nearest',
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        """Executes model fitting over the generators."""
        self.get_base_model()
        self.train_valid_generator()

        # Dynamic step calculations based on your 3000+ new images
        steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        print(f"Starting training loop over {self.config.params_classes} classes...")
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.valid_generator
        )

        # Save the finalized model file locally
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )