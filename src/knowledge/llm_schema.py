from pydantic import BaseModel, Field
from typing import List

class SinglePSARecord(BaseModel):
    english: str = Field(description="The Public Service Announcement (PSA) sentence in English. Must contain exactly one core action directed at the public. Must be concise, exactly 1 sentence, and between 10 to 25 words.")
    domain: str = Field(description="The target domain of the announcement.")
    topic: str = Field(description="The topic of the announcement.")
    subtopic: str = Field(description="The subtopic of the announcement.")
    scenario_id: str = Field(description="The scenario ID.")
    intent: str = Field(description="The communication intent: Advice, Warning, Reminder, Alert, Instruction, Campaign, Notification.")
    severity: str = Field(description="The severity level: Routine, Warning, Emergency.")
    audience: str = Field(description="The specific target citizen audience group (e.g. drivers, farmers, candidates).")
    distribution_channel: str = Field(description="The primary distribution channel: Radio, SMS, Poster, Social Media.")
    tone: str = Field(description="The target tone: Informational, Urgent, Authoritative, Community-outreach.")
    syntactic_pattern: str = Field(description="The syntactic pattern style: Imperative, Declarative, Passive, Conditional.")
    lexical_profile: str = Field(description="The lexical style profile: Plain, Formal, Administrative.")
    word_count: int = Field(description="Number of words in the generated English sentence.")

class PSABatchResponse(BaseModel):
    psas: List[SinglePSARecord] = Field(description="List of generated PSA records.")
