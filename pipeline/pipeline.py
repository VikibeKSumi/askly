from retrieval.retrieval import Retrieve
from generation.generation import Generation
from typing import Any
from pinecone import Pinecone, Document, Index
from langchain_openai import ChatOpenAI
from generation.context_builder import build_context
from pipeline.build_output import build_output

class AsklyRAG():

    def __init__(self,
            vector_client: Pinecone, vector_index: Index,
            dense_model: str, sparse_model: str,
            namespace: str, top_k: int,
            dense_search_field: str, sparse_search_field: str,
            rerank_top_n: int, rrf_weights: list[float],
            rrf_k: int, rrf_top_n: int,llm_client: ChatOpenAI,
            llm_model: str, rerank_model: str,
            include_fields: list[str], search_type: str,
            fallback_answer: str, context_build_threshold: float
        ):

        # settings (from config)
        self.llm_model = llm_model
        self.fallback_answer = fallback_answer
        self.context_build_threshold = context_build_threshold

        # dependencies
        self.generate = Generation(
            llm_client=llm_client, llm_model=llm_model
        )

        # components
        self.retrieve = Retrieve(
            vector_client=vector_client, vector_index=vector_index,
            dense_model=dense_model, sparse_model=sparse_model,
            namespace=namespace, top_k=top_k, 
            dense_search_field=dense_search_field, sparse_search_field=sparse_search_field,
            rerank_top_n=rerank_top_n, rrf_weights=rrf_weights,
            rrf_k=rrf_k, rrf_top_n=rrf_top_n,
            rerank_model=rerank_model, include_fields=include_fields,
            search_type=search_type
        )

        

    def run(self, query):

        reranked_results: list[Document]
        retrieval_info: dict[str, Any]
        reranked_results, retrieval_info = self.retrieve.retrieve(
            query=query
        )

        context_prompt, sources, contexts = build_context(
            reranked_results=reranked_results, context_build_threshold=self.context_build_threshold)
        if not sources:  
            fallback_answer = self.fallback_answer                         
            return build_output(
                query=query, retrieval_info=retrieval_info,
                answer=fallback_answer, answered=False,
                llm_model=None, sources=[],
                contexts=[], llm_ms=0
            )
        
        generated_result: dict[str, dict] = self.generate.generate_response(
            context_prompt=context_prompt,
            query=query
        )

        return build_output(
            query, retrieval_info, answer=generated_result.get("answer"),
            answered=generated_result.get("answered"), llm_model=self.llm_model,
            sources=sources, contexts=contexts, llm_ms=generated_result.get("llm_ms")
        )
