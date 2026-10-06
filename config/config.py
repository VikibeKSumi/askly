import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AsklyConfig():

    def __init__(self, file_path: Path):
        with open(file_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        
        self.DENSE_MODEL_NAME = config_data["models"]["dense_model_name"]
        self.SPARSE_MODEL_NAME = config_data["models"]["sparse_model_name"]
        self.RERANK_MODEL_NAME = config_data["models"]["rerank_model_name"]
        self.INDEX_NAME = config_data["vector_db"]["index_name"]
        self.NAMESPACE = config_data["vector_db"]["namespace"]
        self.TOP_K =  config_data["vector_search"]["top_k"]
        self.RRF_WEIGHTS = config_data["vector_search"]["rrf_weights"]
        self.RRF_K = config_data["vector_search"]["rrf_k"]
        self.RRF_TOP_N = config_data["vector_search"]["rrf_top_n"]
        self.RERANK_TOP_N = config_data["vector_search"]["rerank_top_n"]
        self.INCLUDE_FIELDS = config_data["vector_search"]["include_fields"]
        self.DENSE_SEARCH_FIELD = config_data["vector_search"]["dense_search_field"]
        self.SPARSE_SEARCH_FIELD = config_data["vector_search"]["sparse_search_field"]
      
    
try: 
    file_path = Path(__file__).parent / "config.yaml"
    config = AsklyConfig(file_path=file_path)
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.exception(f"An error occured in conifguration loading stage: {e}")
    raise