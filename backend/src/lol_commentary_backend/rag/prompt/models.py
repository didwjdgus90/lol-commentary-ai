from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROMPT_VERSION = "lol_rag_prompt_v1"


class RagPromptPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    prompt_version: Literal["lol_rag_prompt_v1"] = PROMPT_VERSION

    instructions: str = Field(min_length=1)

    input_text: str = Field(min_length=1)

    evidence_count: int = Field(ge=0)

    citation_ids: tuple[str, ...]
