
from pinecone import Document

def rrf(
        ranked_lists: list[list[Document]],
        weights: list[float],
        k: int,
        top_n: int
    )-> list[Document]:


    scores, seen = {}, {}
    for matches, weight in zip(ranked_lists, weights):
        for rank, match in enumerate(matches, start=1):
            scores[match.id] = scores.get(match.id, 0.0) + weight / (k + rank)
            seen.setdefault(match.id, match)
    
    order = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [seen[mid] for mid in order]    