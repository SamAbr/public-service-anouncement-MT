from pydantic import BaseModel, Field
from typing import List

class SinglePSARecord(BaseModel):
    english: str = Field(description="The Public Service Announcement (PSA) sentence in English. Must contain exactly one core action directed at the public. Must be concise, 1 to 2 sentences (prefer 1 sentence), and between 10 to 25 words.")

class PSABatchResponse(BaseModel):
    psas: List[SinglePSARecord] = Field(description="List of generated PSA records.")
