import os
import json
import time
from typing import List, Dict
from openai import OpenAI

from .knowledge.scenarios import SCENARIOS, INSTITUTIONS, AUDIENCES, HAZARDS, LOCATIONS
from .knowledge.llm_schema import PSABatchResponse, SinglePSARecord
from .validator import ValidationEngine

SYSTEM_PROMPT = """You are an expert copywriter specializing in designing Kenyan Public Service Announcements (PSAs).
Your goal is to generate highly authentic, natural-sounding English PSAs that adhere strictly to the target PSA framework and match the punchy, action-oriented style of real-world Kenyan PSAs.

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

### PSA FRAMEWORK & STYLE CRITERIA (CRITICAL):
1. PUNCHY & DIRECT STYLE: Kenyan PSAs are practical, conversational, and direct. Often they begin with an active exclamation hook like: "Bridge the gap!", "Be prepared!", "Act now!", "Audit schools!", "Access education!".
2. SPECIFIC details: Include realistic specifics such as dates (e.g. "before Nov 30", "by Jan 6"), portals, stipends, or county levels.
3. CONCISENESS: The PSA must be EXACTLY ONE SENTENCE (or occasionally TWO short sentences), and contain between 10 to 25 words.
4. NO EVENT REPORTING: Do not describe government activities, CS meetings, events, launches, or CS visits (REJECT: 'launched', 'CS announced', 'held a meeting').
5. NO GAZETTE/LEGAL NOTICE STYLE: Avoid legalistic or administrative notices (REJECT: 'pursuant to', 'Gazette Notice', 'hereby notifies').
6. PLAIN ENGLISH FOR MT OPTIMIZATION: Avoid idioms, metaphors, or syntactic ambiguity. The sentence must communicate exactly one core action in clear, simple language.
7. ENTITY ALIGNMENT: You must strictly reference the target Institution, Audience, Hazard, and Location provided in the prompt.

### STYLE EXAMPLES FOR ALIGNMENT:
❌ Bad PSA (Press Release Style):
"The Cabinet Secretary launched a new national vaccination drive at the Ministry headquarters to address rising diseases."

❌ Bad PSA (Gazette Notice Style):
"Pursuant to section 5 of the regulations, all motorists are hereby notified to check registration guidelines on speed limits."

✔️ Good Authentic Kenyan PSAs (Routine / Emergency):
- "Bridge the gap! Build more schools in rural areas like Kitui."
- "Bursary application portal now open for needy students in all 47 counties."
- "15,000 full secondary scholarships for arid counties – apply before Nov 30 via county office!"
- "Apply for TVET bursaries by March 15 via HELB portal."
- "All boarding schools must conduct safety audit by September 30, 2025."
- "Heavy rainfall is expected this week. Avoid crossing flooded rivers and follow local safety guidance."
"""

class AzureOpenAIGenerator:
    def __init__(self, api_key: str = None, endpoint: str = None, deployment: str = None, validator: ValidationEngine = None):
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or "https://sagebremariam-4420-resource.services.ai.azure.com/openai/v1"
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "psa-generator"
        self.validator = validator or ValidationEngine(min_words=10, max_words=25)
        
        self.provenance_log_file = "data/provenance_log.json"
        self.provenance_log = []
        
        # Generation stats trackers
        self.total_accepted = 0
        self.total_rejected = 0
        self.total_attempts = 0
        
        self.client = None
        if self.api_key and self.endpoint:
            try:
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
        Generates a batch of validated English PSAs for the scenario, requesting them in a single call to
        minimize latency and API tokens. Automatically runs validators, self-correction loops, and tracks timing.
        """
        if not self.is_configured():
            raise ValueError("Azure AI Foundry Generator is not configured. Please provide base_url endpoint, api_key, and deployment/model name.")

        records = []
        
        user_prompt = f"""Generate {batch_size} unique, distinct, and syntactically varied English PSAs for the following Scenario:

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
- Tone: {scenario_config['tone']}
- Distribution Channel: {scenario_config['distribution_channel']}

### Instructions:
Generate a list of {batch_size} independent public service announcements.
- Speak directly to {scenario_config['audience']} but do NOT explicitly write their group name (like "members of the public", "candidates", "farmers", etc.) inside the sentence as a tag or suffix. The audience should be natural and implied by context.
- Command a public action.
- Use plain, simple English suitable for translation.
- Contain between 10 to 25 words.
- STRICT NEGATIVE CONSTRAINT: Do NOT start any sentence with "If", "If you", "If eligible", or "If facing". 
- STRICT NEGATIVE CONSTRAINT: Do NOT append or embed trailing vocative tags (e.g. reject: "..., members of the public" or "..., candidates").
- STRICT STYLE RULE: Write punchy, active sentences that start directly with an imperative command (e.g. "Apply for...", "Verify your...", "Report...", "Avoid...") or a direct statement (e.g. "All students get..."). Do NOT use passive gerund subjects (e.g. "registering is advised").
"""

        validation_history = []
        candidates_to_fill = batch_size
        attempts = 0
        max_attempts = 3
        
        while len(records) < batch_size and attempts < max_attempts:
            attempts += 1
            self.total_attempts += 1
            
            # 1. Build the prompt
            if attempts == 1:
                prompt_str = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nReturn output in valid JSON format matching this schema: {{\"psas\": [{{\"english\": \"PSA text\"}}]}}"
            else:
                # Compile validation errors for feedback loop
                failed_items_info = []
                for hist in validation_history:
                    if not hist["passed"] and hist["attempt"] == attempts - 1:
                        failed_items_info.append(f"Failed candidate: '{hist['text']}' -> Reason: {hist['failure_reason']}")
                errors_str = "\n".join(failed_items_info)
                
                prompt_str = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\n### Previous Attempt Errors:\n{errors_str}\n\nPlease generate {candidates_to_fill} new/corrected PSAs that completely satisfy the rules. Ensure they are 10-25 words, command action, and have no legal/press keywords. Return output in valid JSON format matching this schema: {{\"psas\": [{{\"english\": \"PSA text\"}}]}}"

            # 2. Call Azure Responses API with timing
            t0 = time.time()
            try:
                response = self.client.responses.create(
                    model=self.deployment,
                    input=prompt_str
                )
                raw_text = response.output_text
            except Exception as e:
                print(f"API Error: {e}")
                break
                
            api_duration = time.time() - t0
            print(f"[Attempt {attempts}] API Call completed in {api_duration:.2f} seconds.")

            # 3. Clean and parse JSON
            t1 = time.time()
            try:
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
                candidates = parsed.psas if parsed and parsed.psas else []
            except Exception as e:
                print(f"JSON Parsing Error: {e}")
                candidates = []
                
            # 4. Run multi-stage validations with timing
            for candidate in candidates:
                candidate_text = candidate.english
                is_valid, reason = self.validator.validate(candidate_text)
                
                if is_valid:
                    if not self.validator.is_semantically_unique(candidate_text, threshold=0.92):
                        is_valid = False
                        reason = "Similarity overlap check failed"
                        
                validation_history.append({
                    "attempt": attempts,
                    "text": candidate_text,
                    "passed": is_valid,
                    "failure_reason": None if is_valid else reason
                })
                
                if is_valid:
                    self.total_accepted += 1
                    records.append({
                        "English": candidate_text,
                        "intent": scenario_config['intent'],
                        "severity": scenario_config['severity'],
                        "syntactic_pattern": scenario_config['syntactic_pattern'],
                        "lexical_profile": scenario_config['lexical_profile'],
                        "word_count": len(candidate_text.split())
                    })
                else:
                    self.total_rejected += 1
                    
            validation_duration = time.time() - t1
            print(f"[Attempt {attempts}] Validation completed in {validation_duration:.4f} seconds.")
            
            # Recalculate remaining candidates needed
            candidates_to_fill = batch_size - len(records)
            if candidates_to_fill <= 0:
                break
                
        # Log provenance details
        psa_id_base = f"PSA_{scenario_config['domain'].split()[0][:3].upper()}"
        for idx, record in enumerate(records):
            psa_id = f"{psa_id_base}_{len(self.provenance_log) + 1:05d}"
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
                "validation_history": [h for h in validation_history if h["text"] == record["English"] or not h["passed"]],
                "passed": True
            }
            self.provenance_log.append(log_entry)
            
        self.save_provenance_log()
        
        # Log batch run summary metrics
        total_attempts = self.total_attempts
        avg_attempts = total_attempts / max(1, self.total_accepted)
        print(f"=== BATCH METRICS: Accepted: {self.total_accepted} | Rejected: {self.total_rejected} | Avg Attempts per PSA: {avg_attempts:.2f} ===")
        
        return records[:batch_size]

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
