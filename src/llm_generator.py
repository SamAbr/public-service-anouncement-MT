import os
import json
import random
from typing import List, Dict
from openai import AzureOpenAI

from .knowledge.scenarios import SCENARIOS, INSTITUTIONS, AUDIENCES, HAZARDS, LOCATIONS
from .knowledge.llm_schema import PSABatchResponse, SinglePSARecord
from .validator import ValidationEngine

SYSTEM_PROMPT = """You are an expert copywriter specializing in designing Kenyan Public Service Announcements (PSAs).
Your goal is to generate highly authentic, natural-sounding English PSAs that adhere strictly to the target PSA framework.

### PSA FRAMEWORK CRITERIA (CRITICAL RULES):
1. CORE INTENT: The PSA must command the public to DO something (take an action, avoid something, or be alert).
2. CONCISENESS: The PSA must be EXACTLY ONE SENTENCE, and contain between 10 to 25 words.
3. NO EVENT REPORTING: Do not describe government activities, CS meetings, events, launches, or statements (REJECT styles that use: 'launched', 'inaugurated', 'Cabinet Secretary announced', 'held a meeting today', 'welcomed delegates').
4. NO GAZETTE/LEGAL NOTICE STYLE: Avoid legalistic or administrative layouts (REJECT styles that use: 'pursuant to section', 'in exercise of the powers', 'Gazette Notice', 'tender No', 'failure to comply', 'hereby notifies').
5. PLAIN ENGLISH FOR MT OPTIMIZATION: Avoid idioms, metaphors, or syntactic ambiguity. The sentence must communicate exactly one core action directed at the specified audience group in clear, simple language.
6. ENTITY ALIGNMENT: You must strictly reference the target Institution, Audience, Hazard, and Location provided in the prompt.
"""

class AzureOpenAIGenerator:
    def __init__(self, api_key: str = None, endpoint: str = None, deployment: str = None, validator: ValidationEngine = None):
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.validator = validator or ValidationEngine(min_words=10, max_words=25)
        
        self.client = None
        if self.api_key and self.endpoint:
            try:
                self.client = AzureOpenAI(
                    api_key=self.api_key,
                    api_version="2024-08-01-preview",  # Version supporting Structured Outputs
                    azure_endpoint=self.endpoint
                )
            except Exception as e:
                print(f"Warning: Failed to initialize Azure OpenAI client: {e}")

    def is_configured(self) -> bool:
        return self.client is not None and self.deployment is not None

    def generate_batch(self, scenario_config: Dict, batch_size: int = 5) -> List[Dict]:
        """
        Calls Azure OpenAI to generate a batch of structured PSAs matching the planned scenario configuration.
        """
        if not self.is_configured():
            raise ValueError("Azure OpenAI Generator is not configured. Please provide endpoint, api_key, and deployment name.")

        user_prompt = f"""Generate a batch of {batch_size} unique, distinct, and syntactically varied English PSAs for the following Scenario:

### Scenario Config:
- Domain: {scenario_config['domain']}
- Topic: {scenario_config['topic']}
- Subtopic: {scenario_config['subtopic']}
- Scenario ID: {scenario_config['scenario_id']}
- Target Institution: {scenario_config['institution']}
- Target Audience: {scenario_config['audience']}
- Allowed Hazard: {scenario_config['hazard']}
- Location Context: {scenario_config['location']}
- Intent: {scenario_config['intent']}
- Severity: {scenario_config['severity']}
- Syntactic Pattern style (mix/match in sentences): {scenario_config['syntactic_pattern']}
- Lexical Profile: {scenario_config['lexical_profile']}
- Tone: {scenario_config['tone']}
- Distribution Channel: {scenario_config['distribution_channel']}

### Instructions:
Generate a list of {batch_size} public service announcements. 
For each PSA:
- Use plain, simple English suitable for translation.
- Ensure the tone matches '{scenario_config['tone']}' and the structure reflects '{scenario_config['syntactic_pattern']}'.
- Output exactly 1 sentence containing between 10 and 25 words.
- Ensure the action is directed at {scenario_config['audience']}.
"""

        records = []
        try:
            # Call Structured Output API
            completion = self.client.beta.chat.completions.parse(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=PSABatchResponse,
                temperature=0.7
            )
            
            parsed_response = completion.choices[0].message.parsed
            if parsed_response and parsed_response.psas:
                for psa in parsed_response.psas:
                    # Enforce planned metadata rather than what LLM guessed
                    records.append({
                        "English": psa.english,
                        "Domain": scenario_config['domain'],
                        "Topic": scenario_config['topic'],
                        "Subtopic": scenario_config['subtopic'],
                        "Class": "PSA",
                        "scenario_id": scenario_config['scenario_id'],
                        "intent": psa.intent or scenario_config['intent'],
                        "severity": psa.severity or scenario_config['severity'],
                        "syntactic_pattern": psa.syntactic_pattern or scenario_config['syntactic_pattern'],
                        "lexical_profile": psa.lexical_profile or scenario_config['lexical_profile'],
                        "audience": scenario_config['audience_name'],
                        "distribution_channel": scenario_config['distribution_channel'],
                        "tone": scenario_config['tone'],
                        "word_count": len(psa.english.split())
                    })
        except Exception as e:
            print(f"Error during Azure OpenAI API call: {e}")
            
        return records
