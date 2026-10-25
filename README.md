# Askly

Askly is an technical query assistant over internal company documents. It's based on RAG architecture using hybrid-search fused with rrf and reranking for retrieval and grounded response generation with cited soures. 

**Link**: [Askly api is live in production](https://askly-assistant-production-1081188824450.asia-south1.run.app/docs)

## Features
- **Three stages** - separated into three repo: ETL, Ingestion, RAG (check below). Each for independent tasks connected by only their outputs.
- **Structure-aware chunking** - sections are turned to blocks and each block is chunked by its type: table, list, code blocks and paragraph.
- **Hybrid-search** - for queries that uses both natural language and exact keywords.
- **RRF+reranking** - uses rrf to fuse the results of dense and sparse search, then reorders more strongly relevancy using reranking.
- **Modular design** - each component separated by its concerns for clarity in structure.
- **Cited answers** - the generated response is cited with sources that can be used to verify the truthfulness of the response by the user.
- **Evaluation** - evaluated the two stages. Context recall and context precision on the retrieval side and faithfulness and answer relevancy for generation based on RAGAS framework.
- **Golden dataset** - a golden dataset is prepared with pre-filled questions and expected response. Later filled in with retrieved context and actual response in the evaluation pipeline.
- **CI/CD** - deployed to GCP using the services Cloud Run, Cloud Deploy, Cloud Storage, Cloud Build, Secret Manager, IAM Service Accounts with PR checks that builds on merge. It deploys to staging then production through manual approval.

## Eval metrics
| metric | score |
| --- | --- |
| context recall | 0.96 |
| context precision | 0.82 |
| faithfulness | 0.91 |
| answer relevancy | 0.69 |

## Tech stack
- Core - Python, OpenAI, Pydantic, LangChain
- Data - AWS S3 (Documents), Pinecone (Vector records)
- Embeddings - Pinecone hosted embeddings 
- API - FastAPI, Uvicorn
- Evaluation - RAGAS
- Infrastructure - Cloud Run, Cloud Build, Cloud Deploy, IAM Service Accounts, Secret Manager, Docker

## Part of the Askly system

| Repo | Role |
|---|---|
| [askly-data-preparation](https://github.com/VikibeKSumi/askly-data-preparation) | Raw docs → clean JSON (ETL) |
| [askly-ingestion](https://github.com/VikibeKSumi/askly-ingestion) | Clean JSON → chunks → embeddings → Pinecone |
| **askly** (this repo) | RAG app: query → answer |

**Input:** records in Pinecone index `askly-index`, namespace `askly-namespace`, produced by
[askly-ingestion](https://github.com/VikibeKSumi/askly-ingestion).  
**Output:** one response object per query (answer, citations, sources, contexts, run metadata).

## Run
> Set up your chunks for retrieval first from the ingestion repo above.

1. Getting started 
```bash
git clone https://github.com/VikibeKSumi/askly.git
cd askly
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

2. Set `.env' with your keys
```
PINECONE_API_KEY="pc..."
OPENAI_API_KEY="sk-..."
```

3. Run
```bash
python -m run # CLI
python -m eval_run  # Evaluation
uvicorn app:app --reload # API at localhost:8000 
```
## Configuration
Settings are in [`config/config.yaml`](config/config.yaml)
```yaml
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

retrieval_models:
  dense_model_name: "llama-text-embed-v2"   
  sparse_model_name: "pinecone-sparse-english-v0" 
  rerank_model_name: "bge-reranker-v2-m3"

llm:
  model_name: "gpt-4o-mini"
  temperature: 0

generation:
  context_build_threshold: 0.02
  fallback_answer: "I couldn't find this in the company documents."

evaluation:
  judge_llm: "gpt-4o-mini"
  judge_embedding: "text-embedding-3-small"
  golden_dataset_path: "evaluation/data_and_reports/golden_dataset.json"
  save_path: "evaluation/data_and_reports"
```

## Non goals
- **Not a general chatbot** - answers only from company documents; no web search or general knowledge.
- **Not an agent** - a fixed retrieve → generate pipeline, with no tools or multi-step reasoning.
- **Single LLM provider** - OpenAI only, with no model switching.
- **English only.**
- **No per-user document permissions** - every caller can search every document.
- **No real-time document sync** - documents are updated by re-running data prep and ingestion.
- **No UI in v1** - API only (`/docs` for trying it out).

## Known Limitation (broken, missing, or unsafe)
- `answered` flag can be `false` on valid answers that report a conflict between sources.
- Conflict flagging is prompt-based; reliable on direct conflict questions, inconsistent when the conflict is a side detail.
- `parent_id` is stored on chunks, but parent chunks are not, so parent expansion isn't possible yet.
- CI runs a test step, but no unit tests are written yet.
- No eval gate in CI; RAGAS is run manually, so a regression could reach production.
- No monitoring or tracing (latency, cost, error rate).
- Single-turn only: no conversation memory, no router for small talk.
- Data prep and ingestion run manually; document updates require re-running both repos.
- Small eval set (30 questions, 27 scored); the judge model is the same as the generator (`gpt-4o-mini`).
- Public endpoint with no authentication or rate limiting; anyone with the URL can use up API credits.
- `timestamp` in the response object is always empty.
- No maximum query length; very long inputs go straight to the embedding, reranker and LLM calls.
- Prompt-injection defence relies only on the system prompt.
- Pinecone/OpenAI failures surface as raw 500 errors; no custom error handling or timeouts.
- Shared settings (index, namespace, models, field names) are duplicated in the ingestion and askly configs, with no check that they match.
- Multi-part questions can miss relevant chunks (e.g. a remote-work answer missed the stipend section).
- `context_build_threshold` (0.02) was tuned by hand on a few queries.
- Scale-to-zero on Cloud Run means the first request after idle is slow (cold start).
- The API returns full retrieved contexts, so with a public endpoint any caller can read raw document text.
- Refusal behaviour isn't measured; the 3 out-of-scope questions are excluded from RAGAS scoring.
- English only (the sparse model `pinecone-sparse-english-v0` is English-specific).
- Secret version is pinned (`key: '1'`); rotating an API key needs a manifest change and redeploy.
- No rollback procedure practised; recovery would mean re-promoting an older release by hand.

## Roadmap
**Safety**
- Add API key authentication and rate limiting to the endpoint.
- Stop returning full retrieved contexts in the public API response.
- Add a maximum query length.

**Fixes**
- Fix the `answered` flag on conflict answers.
- Fill the `timestamp` field and add latency metrics per stage (retrieval, rerank, LLM).
- Handle Pinecone/OpenAI failures with timeouts and clear error responses.
- Check that shared settings match between the ingestion and askly configs.

**Testing & evaluation**
- Write unit tests for retrieval, fusion, context building and output.
- Add a RAGAS eval gate to CI to block regressions.
- Save per-question scores and run metadata; compare each run against the previous one.
- Measure refusal accuracy; expand the golden dataset; use a stronger judge model.

**Monitoring**
- Add tracing and monitoring (LangSmith) for latency, cost and errors.

**Features**
- Store parent chunks and add parent expansion for fuller context.
- Add a router for small talk vs document questions.
- Add multi-turn conversation memory.
- Automate data prep and ingestion as a pipeline for document updates.
- Add a Streamlit UI.
