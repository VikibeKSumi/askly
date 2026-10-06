from pinecone import Index, SparseValues, Document
from retrieval.fusion import rrf 


class Search():

    def __init__(
            self,
            vector_index: Index, namespace: str,
            top_k: int, include_fields: list[str],
            dense_search_field: str, sparse_search_field: str,
            rrf_weights: list[float], rrf_k: int,
            rrf_top_n: int
        ):

        # settings (from config)
        self.rrf_weights = rrf_weights
        self.rrf_k = rrf_k
        self.rrf_top_n = rrf_top_n
        self.dense_search_field = dense_search_field
        self.sparse_search_field = sparse_search_field
        self.include_fields = include_fields
        self.namespace = namespace
        self.top_k = top_k

        # dependencies (injected clients)
        self.vector_index = vector_index


    def dense_search(
            self,
            query_dense: list[float]
        ) -> list[Document] :

        dense_results = self.vector_index.documents.search(
            namespace=self.namespace,
            top_k=self.top_k,
            score_by=[{
                "type": "dense_vector",
                "field": self.dense_search_field,
                "values": query_dense
            }],
            filter={},
            include_fields=self.include_fields
        ).matches

        return dense_results

    def sparse_search(
            self,
            query_indices: list[int],
            query_values: list[float]
        ) -> list[Document]:

        sparse_results = self.vector_index.documents.search(
            namespace=self.namespace,
            top_k=self.top_k,
            score_by=[{
                "type": "sparse_vector",
                "field": self.sparse_search_field,
                "sparse_values": SparseValues(
                    indices=query_indices,
                    values=query_values
                )
            }],
            filter={},
            include_fields=self.include_fields
        ).matches
        return sparse_results


    def hybrid_search_fusion(
            self,
            query_dense_values: list[float],
            query_sparse_indices: list[int],
            query_sparse_values: list[float],
        ) -> list[Document]:


        dense_results = self.dense_search(
            query_dense=query_dense_values
            )
        sparse_results = self.sparse_search(
            query_indices=query_sparse_indices,
            query_values=query_sparse_values
        )

        ranked_list = [dense_results, sparse_results]
        fusion_result = rrf(
            ranked_lists=ranked_list,
            weights=self.rrf_weights,
            k=self.rrf_k,
            top_n=self.rrf_top_n)

        return fusion_result
