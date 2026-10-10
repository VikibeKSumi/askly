
from evaluation.get_data import load_golden_dataset, build_eval_dataset
from evaluation.eval_service import EvalService
from evaluation.save_evals import save_eval_dataset, save_eval_output
from pipeline.pipeline import AsklyRAG
from pathlib import Path
from openai import AsyncOpenAI
import logging 

logger = logging.getLogger(__name__)

class AsklyEval():

    def __init__(
            self, askly: AsklyRAG,
            judge_llm: str, judge_embedding: str,
            client: AsyncOpenAI
        ):

        self.judge_llm = judge_llm
        self.judge_embedding = judge_embedding
        self.eval_service = EvalService(
            client=client,
            judge_llm=self.judge_llm,
            judge_embedding=self.judge_embedding
        )
        self.askly = askly



    async def run_eval(self, file_path: Path, eval_dataset_name: str, save_path: Path):
        logger.info("Loading golden dataset...")
        golden_dataset = load_golden_dataset(file_path=file_path)

        logger.info("Loading eval dataset...")
        eval_dataset = build_eval_dataset(golden_dataset=golden_dataset, askly=self.askly)

        logger.info("Running ragas calculation...")
        eval_output = await self.eval_service.evaluate(eval_dataset=eval_dataset)

        logger.info("Saving eval dataset...")
        save_eval_dataset(save_path=save_path, eval_dataset_name=eval_dataset_name, eval_dataset=eval_dataset)

        logger.info("Saving eval outputs...")
        save_eval_output(
            save_path=save_path,
            eval_dataset_name=eval_dataset_name, eval_output=eval_output,
            judge_llm=self.judge_llm, judge_embedding=self.judge_embedding
        )
        