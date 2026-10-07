
from pinecone import Document

def build_context(reranked_results: list[Document], context_build_threshold: float) -> tuple[str, list[str], list[str]]:

    contexts = []
    sources = []
    for rerank in reranked_results:
        if rerank.score >= context_build_threshold:
            contexts.append(rerank.text)
            sources.append(rerank.source_uri)

    context_prompt = ""
    for i, context_chunk in enumerate(contexts, start=1):
        text = context_chunk
        context_prompt += f"[{i}] {text}\n\n"

    return  context_prompt, sources, contexts
