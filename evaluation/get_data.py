import json
from pipeline.pipeline import AsklyRAG
import logging
logger = logging.getLogger(__name__)

def load_golden_dataset(
        file_path: str
    ) -> list[dict]:

    with open(file_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    return golden_data
 
def build_eval_dataset(
        golden_dataset: list[dict],
        askly: AsklyRAG
    ) -> list[dict]:

    eval_dataset  = []
    

    for i, data in enumerate(golden_dataset, start=1):
        logger.info(f"Answering question {i}/{len(golden_dataset)}...")
        user_input = data.get("user_input", "")
         
        output = askly.run(query=user_input)
        response = output.get("generation_info").get("answer")
        retrieved_contexts = output.get("contexts")

        eval_dataset .append({
            "user_input": user_input,
            "reference": data.get("reference"),
            "response": response ,
            "retrieved_contexts": retrieved_contexts,
            "is_refusal": data.get("is_refusal", False)
        })

    return eval_dataset  
