from pinecone import Pinecone, Document, Index
from retrieval.embedding import dense_embed, sparse_embed
from retrieval.search import Search
from retrieval.rerank import rerank


class Retrieve():

    def __init__(
            self,
            vector_client: Pinecone, vector_index: Index,
            dense_model: str, sparse_model: str,
            namespace: str, top_k: int,
            dense_search_field: str, sparse_search_field: str,
            rerank_top_n: int, rrf_weights: list[float],
            rrf_k: int, rrf_top_n: int, rerank_model: str,
            include_fields: list[str], search_type: str
        ):

        # settings (from config)
        self.search_type = search_type
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.rerank_model = rerank_model
        self.namespace = namespace
        self.dense_search_field = dense_search_field
        self.sparse_search_field = sparse_search_field
        self.include_fields = include_fields
        self.rrf_weights = rrf_weights
        self.rrf_k = rrf_k
        self.rrf_top_n = rrf_top_n
        self.rerank_top_n = rerank_top_n
        self.top_k = top_k

        # dependencies (injected clients)
        self.vector_client = vector_client
        self.vector_index = vector_index

        # components (build from above)
        self.hybrid_search = Search(
            vector_index =self.vector_index,
            namespace=self.namespace,
            top_k=self.top_k,
            include_fields=self.include_fields,
            dense_search_field=self.dense_search_field,
            sparse_search_field=self.sparse_search_field,
            rrf_weights=self.rrf_weights,
            rrf_k=self.rrf_k,
            rrf_top_n=self.rrf_top_n

        )

    def retrieve(self, query: str) -> tuple[list[Document], dict[str, dict]]:

        query_dense_values: list[float] = dense_embed(
            vector_client=self.vector_client,
            query=query,
            dense_model=self.dense_model
        )
        query_sparse_indices, query_sparse_values = sparse_embed(
            vector_client=self.vector_client,
            query=query,
            sparse_model=self.sparse_model
        )

        hybrid_search_results: list[Document] = self.hybrid_search.hybrid_search_fusion(
            query_dense_values=query_dense_values,
            query_sparse_indices=query_sparse_indices,
            query_sparse_values=query_sparse_values,
        )

        reranked_results: list[Document] = rerank(
            vector_client=self.vector_client,
            rerank_model=self.rerank_model,
            query=query,
            matches=hybrid_search_results,
            rerank_top_n=self.rerank_top_n
        )
        retrieval_info = {
              "retrieval_info": {
                "search_type": self.search_type,
                "top_k": self.top_k,
                "rerank_top_n": self.rerank_top_n,
                "models": {
                    "dense": self.dense_model,
                    "sparse": self.sparse_model,
                    "rerank": self.rerank_model,
                },
                "retrieval_ms": "",
            }
        }

        return reranked_results, retrieval_info
    


