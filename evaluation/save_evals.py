from typing import Any
import json
from pathlib import Path

def save_eval_dataset(
        eval_dataset_name: str, eval_dataset: list[dict[str, Any]],
        save_path: Path
    ) -> None:

    with open(f"{save_path}/{eval_dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, ensure_ascii=False, indent=2)
    
def save_eval_output(
        save_path: Path,
        eval_dataset_name: str, eval_output: dict[str, float],
        judge_llm: str, judge_embedding: str
    ) -> None:
    info = {
        "evaluation_info": {
            "dataset_name": f"{eval_dataset_name}.json",
            "judge_llm": judge_llm,
            "judge_embedding": judge_embedding,
        },
        "scores": eval_output
    }
    with open(f"{save_path}/{eval_dataset_name}_values.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
