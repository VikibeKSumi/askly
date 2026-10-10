import yaml
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

class AsklyConfig():

    def __init__(self, file_path: Path):
        with open(file_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        
        self.DENSE_MODEL_NAME = config_data["retrieval_models"]["dense_model_name"]
        self.SPARSE_MODEL_NAME = config_data["retrieval_models"]["sparse_model_name"]
        self.RERANK_MODEL_NAME = config_data["retrieval_models"]["rerank_model_name"]
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
        self.SEARCH_TYPE = config_data["vector_search"]["search_type"]
        self.LLM_MODEL_NAME = config_data["llm"]["model_name"]
        self.LLM_TEMPERATURE = config_data["llm"]["temperature"]
        self.FALLBACK_ANSWER = config_data["generation"]["fallback_answer"]
        self.CONTEXT_BUILD_THRESHOLD = config_data["generation"]["context_build_threshold"]
        self.JUDGE_LLM = config_data["evaluation"]["judge_llm"]
        self.JUDGE_EMBEDDING = config_data["evaluation"]["judge_embedding"]
        self.GOLDEN_DATASET_PATH = config_data["evaluation"]["golden_dataset_path"]
        self.SAVE_PATH = config_data["evaluation"]["save_path"]


file_path = Path(__file__).parent / "config.yaml"
config = AsklyConfig(file_path=file_path)
logger.info("Configuration loaded successfully")