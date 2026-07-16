import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Kidney_Disease_Classification.config.configuration import ConfigurationManager
from Kidney_Disease_Classification.components.model_evaluation_mlflow import Evaluation
from Kidney_Disease_Classification import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        """Orchestrates configuration instantiation and component evaluation methods."""
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation = Evaluation(eval_config)
        # 1. Run local evaluations & write scores.json
        logger.info("Starting local validation dataset evaluation...")
        evaluation.evaluation()
        
        # 2. Push experiment data out to DagsHub MLflow tracking
        logger.info("Streaming parameters, metrics, and models to DagsHub MLflow Server...")
        evaluation.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e