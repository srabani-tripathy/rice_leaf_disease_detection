from rldd.config import ConfigurationManager
from rldd.components import DataIngestion
from rldd import logger

class DataIngestionTrainingPipeline:
    def __int__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.unzip_and_clean()