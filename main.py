from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation,DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os
import sys
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
if __name__=='__main__':
    try:
        trainignpipelineconfig=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(trainignpipelineconfig)
        data_ingestion=DataIngestion(data_ingestion_config)
        logging.info("Intitiate the data ingestion")
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info("Data Intitation Completed")
        data_validation_config=DataValidationConfig(trainignpipelineconfig)
        data_validation=DataValidation(data_ingestion_artifact,data_validation_config)
        logging.info("Initiate the data Validation")
        data_validation_artifact=data_validation.intiate_data_validation()
        logging.info("data Validation Completed")
        print(data_validation_artifact)
        
    except Exception as e:
        raise NetworkSecurityException(e,sys)