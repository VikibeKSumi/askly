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

