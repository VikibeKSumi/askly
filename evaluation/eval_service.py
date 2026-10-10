import math
from typing import Any
from openai import AsyncOpenAI
from ragas.metrics.collections import (
    Faithfulness ,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision 
)
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory 
from ragas.metrics.result import MetricResult
import logging
logger = logging.getLogger(__name__)

class EvalService():

    def __init__(self,
            client: AsyncOpenAI, judge_llm: str,
            judge_embedding: str
        ):

        self.client = client
        self.judge_llm = llm_factory(model=judge_llm, client=self.client)
        self.judge_embedding = embedding_factory("openai", model=judge_embedding, client=self.client)
        self.faithfulness = Faithfulness(llm=self.judge_llm)
        self.answer_relevancy = AnswerRelevancy(llm=self.judge_llm, embeddings=self.judge_embedding)
        self.context_recall = ContextRecall(llm=self.judge_llm)
        self.context_precision = ContextPrecision(llm=self.judge_llm)


    def average(self, results : list[MetricResult]) -> float:
        values = [r.value for r in results if r.value is not None and not math.isnan(r.value)]
        return round(sum(values) / len(values), 3)


    async def evaluate(self, eval_dataset: list[dict]) -> dict[str, Any]:

        scored = [d for d in eval_dataset if not d.get("is_refusal")]

        logger.info(f"Scoring faithfulness {len(scored)} samples...")
        f_results: list[MetricResult] = await self.faithfulness.abatch_score([
            {"user_input": eval_data["user_input"], "response": eval_data["response"],
            "retrieved_contexts": eval_data["retrieved_contexts"]}
            for eval_data in scored
        ])

        logger.info(f"Scoring answer relevancy {len(scored)} samples...")
        ar_results = await self.answer_relevancy.abatch_score([
            {"user_input": eval_data["user_input"], "response": eval_data["response"]}
            for eval_data in scored
        ])

        logger.info(f"Scoring context recall {len(scored)} samples...")
        cr_results = await self.context_recall.abatch_score([
            {"user_input": eval_data["user_input"], "retrieved_contexts": eval_data["retrieved_contexts"],
            "reference": eval_data["reference"]}
            for eval_data in scored
        ])

        logger.info(f"Scoring context precision {len(scored)} samples...")
        cp_results = await self.context_precision.abatch_score([
            {"user_input": eval_data["user_input"], "reference": eval_data["reference"],
            "retrieved_contexts": eval_data["retrieved_contexts"]}
            for eval_data in scored
        ])

        eval_output = {
            "faithfulness": self.average(f_results),
            "answer_relevancy": self.average(ar_results),
            "context_recall": self.average(cr_results),
            "context_precision": self.average(cp_results)
        }

        return eval_output