
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from generation.prompts import SYSTEM_PROMPT


def render_prompt(context: str, query: str) -> ChatPromptValue: 

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context} \n\n Query:\n{query}")
    ]).invoke(input={"context":context, "query":query})

    return prompt