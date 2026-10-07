from generation.schema import LLMOutput
from typing import Any
from langchain_openai import ChatOpenAI

class LLMService():

    def __init__(self, llm_client: ChatOpenAI):
        self.llm_structured = llm_client.with_structured_output(LLMOutput)


    def call(self, prompt) -> dict[str, Any]:
        
        response: LLMOutput = self.llm_structured.invoke(input=prompt)
        return {
            "response": response,
            "llm_ms": "",
        }