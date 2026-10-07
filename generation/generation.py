

from langchain_openai import ChatOpenAI
from generation.llm_service import LLMService
from generation.prompt_render import render_prompt
from typing import Any

class Generation():

    def __init__(
            self,
            llm_client: ChatOpenAI, llm_model: str
        ):

        self.llm_model = llm_model
        self.llm_service = LLMService(
            llm_client=llm_client,
        )


    def generate_response(self, context_prompt:str, query: str) -> dict[str, Any]:

        
        prompt_input = {
            "context": context_prompt,
            "query": query
        }
        prompt = render_prompt(**prompt_input)

        response: dict[str, Any] = self.llm_service.call(prompt=prompt)

        return {
            "answer": response.get("response").answer,
            "answered": response.get("response").answered,
            "model": {
                "llm": self.llm_model
            },
            "llm_ms": response.get("llm_ms")
        }