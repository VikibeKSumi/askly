
from pinecone import Pinecone, Document

def rerank(
        vector_client: Pinecone,
        rerank_model: str,
        query: str,
        matches: list[Document],
        rerank_top_n: int,
    ) -> list[Document]:

    by_id = {match.id: match for match in matches}
    results = vector_client.inference.rerank(
        model=rerank_model,
        query=query,
        documents=[{"id": match.id, "chunk_text": match.text} for match in matches],
        rank_fields=["chunk_text"],
        top_n=rerank_top_n,
        return_documents=True,
        parameters={"truncate": "END"},
    )
    
    return [by_id[result.document["id"]] for result in results.data]   # reranked matches, best first
