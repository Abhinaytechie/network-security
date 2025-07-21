import os
import sys
import shutil

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from huggingface_hub import Repository,HfApi
api_token = os.getenv("HF_TOKEN")

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.model_dir = os.path.abspath("final_model")  # directory with model.pkl and others
        self.repo_id = "Abhinaytechie/network-security"  # Your HF repo ID
        self.repo_local_path = os.path.abspath("huggingface_upload/network-security")
  # Replace with your repo name

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            logging.info("Initiate the data Validation")
            data_validation_artifact = data_validation.intiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def sync_saved_model_dir_to_huggingface(self):
        try:
            # Clean existing local clone
            if os.path.exists(self.repo_local_path):
                for root, dirs, files in os.walk(self.repo_local_path, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except PermissionError:
                            logging.warning(f"Permission denied while deleting file: {name}")
                    for name in dirs:
                        path = os.path.join(root, name)
                        if name != ".git":
                            try:
                                shutil.rmtree(path)
                            except PermissionError:
                                logging.warning(f"Permission denied while deleting folder: {path}")

            # Clone the remote HF repo to local
            repo = Repository(
                local_dir=self.repo_local_path,
                clone_from=self.repo_id,
                use_auth_token=api_token
            )

            # Copy both model.pkl and preprocess.pkl
            dst_model_path = os.path.join(self.repo_local_path, "model")
            os.makedirs(dst_model_path, exist_ok=True)
            for file_name in ["model.pkl", "preprocessor.pkl"]:
                src = os.path.join(self.model_dir, file_name)
                dst = os.path.join(dst_model_path, file_name)
                shutil.copy2(src, dst)

            # Push to Hugging Face
            repo.push_to_hub(commit_message="Uploaded model and preprocessor for prediction")

            logging.info(f"✅ Uploaded to: https://huggingface.co/{self.repo_id}")

        except Exception as e:
            logging.error("❌ Failed to upload model/preprocessor to Hugging Face")
            raise NetworkSecurityException(e,sys)

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)

            self.sync_saved_model_dir_to_huggingface()

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
