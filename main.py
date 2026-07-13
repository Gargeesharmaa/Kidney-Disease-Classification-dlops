import os
import sys

# Point Python to the 'src' directory directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Now these imports will work perfectly!
from Kidney_Disease_Classification import logger
from Kidney_Disease_Classification.pipeline.stage1_data_ingestion import DataIngestionTrainingPipeline

from Kidney_Disease_Classification import logger
from Kidney_Disease_Classification.pipeline.stage1_data_ingestion import DataIngestionTrainingPipeline



STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
