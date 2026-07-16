import os
from pathlib import Path
from Kidney_Disease_Classification.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from Kidney_Disease_Classification.utils.common import read_yaml, create_directories,save_json
from Kidney_Disease_Classification.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig, EvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):


        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    


    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LR,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        data_ingestion = self.config.data_ingestion
        
        create_directories([Path(training.root_dir)])
        
        # Point dynamically to the unzipped 4-class image directory folder
        training_data_dir = os.path.join(data_ingestion.unzip_dir, "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(training.updated_base_model_path),
            training_data=Path(training_data_dir),
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
            params_classes=self.params.CLASSES 
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        # 1. Fetch paths from config.yaml mapping
        # We evaluate the final trained model produced by Stage 3
        path_of_model = self.config.training.trained_model_path
        
        # 2. Dynamically locate the dataset path (pointing to your 4 folders)
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")
        
        # 3. Package the configuration values into the Entity class
        eval_config = EvaluationConfig(
            path_of_model=Path(path_of_model),
            training_data=Path(training_data),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_epochs=self.params.EPOCHS,
            params_classes=self.params.CLASSES
        )
        
        return eval_config
