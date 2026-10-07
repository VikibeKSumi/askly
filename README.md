# askly



## Part of the Askly system

| Repo | Role |
|---|---|
| [askly-data-preparation](https://github.com/VikibeKSumi/askly-data-preparation) | Raw docs → clean JSON (ETL) |
| [askly-ingestion](https://github.com/VikibeKSumi/askly-ingestion) | Clean JSON → chunks → embeddings → Pinecone |
| **askly** (this repo) | RAG app: query → answer |

**Input:** records in Pinecone index `askly-index`, namespace `askly-namespace`, produced by
[askly-ingestion](https://github.com/VikibeKSumi/askly-ingestion).
**Output:** one response object per query (answer, citations, sources, contexts, run metadata).


## Configuration
Settings are in [`config/config.yaml`](config/config.yaml)
```yaml
models:
  dense_model_name: "llama-text-embed-v2"   
  sparse_model_name: "pinecone-sparse-english-v0" 
  rerank_model_name: "bge-reranker-v2-m3"

vector_db: #pinecone
  index_name: "askly-index"
  namespace: "askly-namespace"

vector_search:
  search_type: "hybrid+rerank"
  #retrieval
  top_k: 10

  # fusion
  rrf_weights: [0.7, 0.3]
  rrf_k: 60
  rrf_top_n: 10

  # rerank
  rerank_top_n: 5

  # field setting
  include_fields: ["text", "parent_id", "section", "doc_id", "title", "source_uri"]
  dense_search_field: "embedding"
  sparse_search_field: "sparse_value"

llm:
  model_name: "gpt-4o-mini"
  temperature: 0

generation:
  context_build_threshold: 0.5
  fallback_answer: "I couldn't find this in the company documents."

```