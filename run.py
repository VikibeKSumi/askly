import logging
from rich.logging import RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(message)s",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import ChatOpenAI

from config.config import config
from pipeline.pipeline import AsklyRAG


load_dotenv()

if __name__ == "__main__":
    try:
        INDEX_NAME = config.INDEX_NAME
        DENSE_MODEL_NAME = config.DENSE_MODEL_NAME
        SPARSE_MODEL_NAME = config.SPARSE_MODEL_NAME
        NAMESPACE = config.NAMESPACE
        TOP_K = config.TOP_K
        DENSE_SEARCH_FIELD = config.DENSE_SEARCH_FIELD
        SPARSE_SEARCH_FIELD = config.SPARSE_SEARCH_FIELD
        RERANK_TOP_N = config.RERANK_TOP_N
        RRF_WEIGHTS = config.RRF_WEIGHTS
        RRF_K = config.RRF_K
        RRF_TOP_N = config.RRF_TOP_N
        RERANK_MODEL_NAME = config.RERANK_MODEL_NAME
        INCLUDE_FIELDS = config.INCLUDE_FIELDS
        SEARCH_TYPE = config.SEARCH_TYPE
        LLM_MODEL_NAME = config.LLM_MODEL_NAME
        LLM_TEMPERATURE = config.LLM_TEMPERATURE
        FALLBACK_ANSWER = config.FALLBACK_ANSWER
        CONTEXT_BUILD_THRESHOLD = config.CONTEXT_BUILD_THRESHOLD

        vector_client = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )
        vector_index = vector_client.Index(
            name=INDEX_NAME
        )

        llm_client = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE
        )

        askly = AsklyRAG(
            vector_client=vector_client,
            vector_index=vector_index,
            dense_model=DENSE_MODEL_NAME,
            sparse_model=SPARSE_MODEL_NAME,
            namespace=NAMESPACE,
            top_k=TOP_K,
            dense_search_field=DENSE_SEARCH_FIELD,
            sparse_search_field=SPARSE_SEARCH_FIELD,
            rerank_top_n=RERANK_TOP_N,
            rrf_weights=RRF_WEIGHTS,
            rrf_k=RRF_K,
            rrf_top_n=RRF_TOP_N,
            rerank_model=RERANK_MODEL_NAME,
            include_fields=INCLUDE_FIELDS,
            search_type=SEARCH_TYPE,
            llm_client=llm_client,
            llm_model=LLM_MODEL_NAME,
            fallback_answer=FALLBACK_ANSWER,
            context_build_threshold=CONTEXT_BUILD_THRESHOLD
        )


        while True:
            query = input("Enter query: ").strip()

            if not query:
                continue
            if query.lower().strip() in ["exit", "quit"]:
                break
            
            response = askly.run(query=query)
            print(response.get("generation_info").get("answer"))

    except Exception as e:
        logger.error(f"An error occured: {e} ")