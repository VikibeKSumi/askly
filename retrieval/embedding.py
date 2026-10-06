
from pinecone import Pinecone, DenseEmbedding, SparseEmbedding

def dense_embed(vector_client: Pinecone, query: str, dense_model: str) -> list[float]:

    embedding = vector_client.inference.embed(
        model=dense_model,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"}
    )[0].values

    return embedding


def sparse_embed(vector_client: Pinecone, query: str, sparse_model: str) -> tuple[list[int], list[float]]:

    embedding = vector_client.inference.embed(
        model=sparse_model,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"}
    )[0]

    sparse_indices = embedding.sparse_indices
    sparse_values = embedding.sparse_values
    return sparse_indices, sparse_values
    

