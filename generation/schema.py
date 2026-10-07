from pydantic import BaseModel, Field

class LLMOutput(BaseModel):
    answer: str = Field(
        description=(
            "The answer to the user's question, using only information from the provided sources. "
            "Cite every fact with its source number in square brackets, e.g. [1] or [2][3]. "
            "If sources disagree, state both versions and cite each. "
            "If the sources do not contain the answer, say so briefly instead of guessing."
        )
    )
    answered: bool = Field(
        description=(
            "True if the answer is supported by the provided sources. "
            "False if the sources do not contain enough information to answer the question."
        )
    )
