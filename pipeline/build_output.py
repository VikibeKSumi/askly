from typing import Any

def build_output(query: str, retrieval_info: dict[str, Any],
                 answer: str, answered: bool, llm_model: str | None, llm_ms: int,
                 sources: list[dict], contexts: list[str]) -> dict[str, Any]:
    return {
        "query": query,
        **retrieval_info,
        "generation_info": {
            "answer": answer,
            "answered": answered,
            "model": {"llm": llm_model},
            "llm_ms": llm_ms,
        },
        "sources": sources,
        "contexts": contexts,
        "timestamp": "",
    }
