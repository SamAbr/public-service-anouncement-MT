import os
import json
from typing import List, Dict
from openai import OpenAI

from .knowledge.scenarios import SCENARIOS, INSTITUTIONS, AUDIENCES, HAZARDS, LOCATIONS
from .knowledge.llm_schema import PSABatchResponse, SinglePSARecord
from .validator import ValidationEngine

SYSTEM_PROMPT = """You are an expert copywriter specializing in designing Kenyan Public Service Announcements (PSAs).
Your goal is to generate highly authentic, natural-sounding English PSAs that adhere strictly to the target PSA framework and reflect the local Kenyan public service context.

### KENYAN ORGANIZATIONS CONTEXT:
Use and reference the correct Kenyan institutions when prompted:
- KNEC (Kenya National Examinations Council): Coordinates school exams (KCSE) and integrity.
- HELB (Higher Education Loans Board): Manages university student loans and bursary disbursements.
- TSC (Teachers Service Commission): Employs and registers public school teachers.
- KUCCPS (Kenya Universities and Colleges Central Placement Service): Handles university course placement.
- MoE (Ministry of Education): Directs schools and educational safety.
- MoA (Ministry of Agriculture): Handles farming advisories and livestock extension support.
- KEPHIS (Kenya Plant Health Inspectorate Service): Inspections, locust/fall armyworm pest control, and seed quality.
- KALRO (Kenya Agricultural and Livestock Research Organisation): Agricultural research.
- NDMA (National Drought Management Authority): Manages drought relief and food security warnings.
- NTSA (National Transport and Safety Authority): Coordinates road safety, vehicle inspections, and speed limits.
- NPS (National Police Service) & DCI (Directorate of Criminal Investigations): Public security, crime reporting.
- NC4 (National Computer and Cybercrimes Coordination Committee): Cybersecurity alerts.
- KRA (Kenya Revenue Authority): Handles iTax returns, taxation compliance, and deadlines.
- EACC (Ethics and Anti-Corruption Commission): Integrities and reporting bribery.
- ODPC (Office of the Data Protection Commissioner): Privacy breaches, data safety.
- MoH (Ministry of Health): Handles disease outbreaks, cholera, polio vaccination campaigns.
- SHA (Social Health Authority): Universal health insurance registrations (formerly NHIF).
- PPB (Pharmacy and Poisons Board): Regulates drug safety and retail chemist licenses.

### PSA FRAMEWORK CRITERIA (CRITICAL RULES):
1. CORE INTENT: The PSA must command the public to DO something (take an action, avoid something, or be alert).
2. CONCISENESS: The PSA must be EXACTLY ONE SENTENCE (or occasionally TWO short sentences), and contain between 10 to 25 words.
3. LOCAL KENYAN GEOGRAPHY & CONTEXT: Frame location context using Kenyan terms like "counties", "sub-counties", "Huduma Centers countrywide", "police stations", or "local chemist outlets".
4. NO EVENT REPORTING: Do not describe government activities, CS meetings, events, launches, or CS visits (REJECT: 'launched', 'CS announced', 'held a meeting').
5. NO GAZETTE/LEGAL NOTICE STYLE: Avoid legalistic or administrative notices (REJECT: 'pursuant to', 'Gazette Notice', 'hereby notifies').
6. PLAIN ENGLISH FOR MT OPTIMIZATION: Avoid idioms, metaphors, or syntactic ambiguity. The sentence must communicate exactly one core action in clear, simple language.
7. ENTITY ALIGNMENT: You must strictly reference the target Institution, Audience, Hazard, and Location provided in the prompt.

### EXAMPLES FOR STYLE GUIDANCE:
❌ Bad PSA (Press Release Style):
"The Cabinet Secretary launched a new national vaccination drive at the Ministry headquarters to address rising diseases."
(Reason: Rejects because it describes a government event/launch rather than directly instructing public action).

❌ Bad PSA (Gazette Notice Style):
"Pursuant to section 5 of the regulations, all motorists are hereby notified to check registration guidelines on speed limits."
(Reason: Rejects because it sounds like a legalistic Gazette Notice).

✔️ Good PSA (Routine Advice):
"Clean your hands thoroughly before preparing food to protect your family from waterborne disease outbreaks."
(Reason: Direct, concise, commands a clear public action, 14 words).

✔️ Good PSA (Emergency Warning - Two Sentences Allowed):
"Heavy rainfall is expected this week. Avoid crossing flooded rivers and follow local safety guidance."
(Reason: Two short sentences, urgent, commands action, 16 words).
"""

class AzureOpenAIGenerator:
    def __init__(self, api_key: str = None, endpoint: str = None, deployment: str = None, validator: ValidationEngine = None):
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        # Default endpoint and deployment matching the Azure AI Foundry configuration
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or "https://sagebremariam-4420-resource.services.ai.azure.com/openai/v1"
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "psa-generator"
        self.validator = validator or ValidationEngine(min_words=10, max_words=25)
        
        self.provenance_log_file = "output/provenance_log.json"
        self.provenance_log = []
        
        self.client = None
        if self.api_key and self.endpoint:
            try:
                # Use OpenAI client with base_url instead of AzureOpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.endpoint
                )
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")

    def is_configured(self) -> bool:
        return self.client is not None and self.deployment is not None

    def generate_batch(self, scenario_config: Dict, batch_size: int = 5) -> List[Dict]:
        """
        Generates a batch of validated English PSAs for the scenario, calling Azure AI Foundry's Responses API,
        incorporating self-correction rewrite loops and provenance logging.
        """
        if not self.is_configured():
            raise ValueError("Azure AI Foundry Generator is not configured. Please provide base_url endpoint, api_key, and deployment/model name.")

        records = []
        
        for i in range(batch_size):
            user_prompt = f"""Generate a unique, natural-sounding English PSA for the following Scenario:

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
- Syntactic Pattern: {scenario_config['syntactic_pattern']}
- Lexical Profile: {scenario_config['lexical_profile']}
- Tone: {scenario_config['tone']}
- Distribution Channel: {scenario_config['distribution_channel']}

### Instructions:
Generate exactly 1 PSA sentence (or 2 short sentences). It must be directed at {scenario_config['audience']}, command a public action, use simple plain English, and contain 10-25 words. Do not include metadata.
"""

            validation_history = []
            final_text = None
            passed = False
            
            for attempt in range(1, 4):
                try:
                    if attempt == 1:
                        prompt_str = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nReturn output in valid JSON format matching this schema: {{\"psas\": [{{\"english\": \"PSA text\"}}]}}"
                    else:
                        last_failed = validation_history[-1]["text"]
                        last_reason = validation_history[-1]["failure_reason"]
                        prompt_str = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nPrevious attempt: '{last_failed}'\nFailed validation because: {last_reason}\n\nPlease rewrite the PSA to fix these issues. Ensure it contains 10-25 words, commands action, and has no legal/press keywords. Return output in valid JSON format matching this schema: {{\"psas\": [{{\"english\": \"PSA text\"}}]}}"
                        
                    # Call Responses API
                    response = self.client.responses.create(
                        model=self.deployment,
                        input=prompt_str
                    )
                    
                    raw_text = response.output_text
                    
                    # Clean markdown code block wraps if present
                    cleaned_text = raw_text.strip()
                    if cleaned_text.startswith("```"):
                        lines = cleaned_text.splitlines()
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines[-1].startswith("```"):
                            lines = lines[:-1]
                        cleaned_text = "\n".join(lines).strip()
                        
                    data = json.loads(cleaned_text)
                    parsed = PSABatchResponse.model_validate(data)
                    
                    if parsed and parsed.psas:
                        candidate_text = parsed.psas[0].english
                        
                        is_valid, reason = self.validator.validate(candidate_text)
                        
                        if is_valid:
                            if not self.validator.is_semantically_unique(candidate_text, threshold=0.92):
                                is_valid = False
                                reason = "Rejected by semantic similarity check (similarity > 0.92)"
                                
                        validation_history.append({
                            "attempt": attempt,
                            "text": candidate_text,
                            "passed": is_valid,
                            "failure_reason": None if is_valid else reason
                        })
                        
                        if is_valid:
                            final_text = candidate_text
                            passed = True
                            break
                    else:
                        validation_history.append({
                            "attempt": attempt,
                            "text": "",
                            "passed": False,
                            "failure_reason": "Parsed list is empty"
                        })
                except Exception as e:
                    validation_history.append({
                        "attempt": attempt,
                        "text": "",
                        "passed": False,
                        "failure_reason": f"API Exception: {str(e)}"
                    })
            
            # Log provenance details
            psa_id = f"PSA_{scenario_config['domain'].split()[0][:3].upper()}_{len(records) + 1:05d}"
            log_entry = {
                "psa_id": psa_id,
                "planner_inputs": {
                    "domain": scenario_config['domain'],
                    "topic": scenario_config['topic'],
                    "subtopic": scenario_config['subtopic'],
                    "scenario_id": scenario_config['scenario_id'],
                    "intent": scenario_config['intent'],
                    "severity": scenario_config['severity'],
                    "syntactic_pattern": scenario_config['syntactic_pattern'],
                    "lexical_profile": scenario_config['lexical_profile'],
                    "audience": scenario_config['audience'],
                    "tone": scenario_config['tone'],
                    "distribution_channel": scenario_config['distribution_channel']
                },
                "raw_prompt": user_prompt,
                "validation_history": validation_history,
                "passed": passed
            }
            self.provenance_log.append(log_entry)
            
            if passed and final_text:
                records.append({
                    "English": final_text,
                    "intent": scenario_config['intent'],
                    "severity": scenario_config['severity'],
                    "syntactic_pattern": scenario_config['syntactic_pattern'],
                    "lexical_profile": scenario_config['lexical_profile'],
                    "word_count": len(final_text.split())
                })
                
        self.save_provenance_log()
        
        return records

    def save_provenance_log(self):
        """
        Saves the generation artifacts log to disk.
        """
        os.makedirs(os.path.dirname(self.provenance_log_file), exist_ok=True)
        try:
            with open(self.provenance_log_file, "w", encoding="utf-8") as f:
                json.dump(self.provenance_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save provenance log file: {e}")
