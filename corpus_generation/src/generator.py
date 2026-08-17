import os
import json
import random
from collections import defaultdict
from .config import DOMAINS, MIN_WORDS, MAX_WORDS, CHECKPOINT_FILE
from .validator import ValidationEngine

# Import scenarios and entities
from .knowledge.scenarios import SCENARIOS, INSTITUTIONS, AUDIENCES, HAZARDS, LOCATIONS
from .knowledge.entities import Context
from .templates.families import TEMPLATE_FAMILIES

class PSAGenerator:
    def __init__(self, size=50000, checkpoint_file=None, engine="azure_llm",
                 azure_api_key=None, azure_endpoint=None, azure_deployment=None, start_counts=None):
        self.size = size
        self.target_per_domain = size // len(DOMAINS)
        self.validator = ValidationEngine(min_words=10, max_words=25)
        self.checkpoint_file = checkpoint_file if checkpoint_file else CHECKPOINT_FILE
        self.engine = "azure_llm"
        self.start_counts = start_counts or {}
        
        from .llm_generator import AzureOpenAIGenerator
        self.llm_generator = AzureOpenAIGenerator(
            api_key=azure_api_key,
            endpoint=azure_endpoint,
            deployment=azure_deployment,
            validator=self.validator
        )
        print("Initialized Azure OpenAI generator backend.")
            
        # Balancing Controller state
        self.stats = {
            "domain": defaultdict(int),
            "scenario_id": defaultdict(int),
            "intent": defaultdict(int),
            "severity": defaultdict(int),
            "syntactic_pattern": defaultdict(int),
            "tone": defaultdict(int),
            "distribution_channel": defaultdict(int)
        }

    def _select_balanced_choice(self, choices, stat_category):
        """
        Implements the Corpus-Level Balancing Controller.
        Selects the candidate with the lowest generation count so far.
        """
        if not choices:
            return None
        scored_choices = [(choice, self.stats[stat_category][str(choice)]) for choice in choices]
        min_freq = min(score for _, score in scored_choices)
        candidates = [choice for choice, score in scored_choices if score == min_freq]
        return random.choice(candidates)

    def generate_english_psas(self):
        """
        Generates the target number of unique, valid English PSAs using concurrent Azure OpenAI (GPT-4o) backend.
        """
        if not self.llm_generator or not self.llm_generator.is_configured():
            raise ValueError(
                "Azure OpenAI is not configured. Please supply an API key and deployment endpoint."
            )

        english_records = []
        batch_size = 20
        tasks = []
        
        for domain in DOMAINS:
            scenarios = SCENARIOS.get(domain, [])
            if not scenarios:
                continue
            
            domain_prefix = domain.split()[0][:3].upper()
            count = 0
            while count < self.target_per_domain:
                # Select scenario & build config
                scenario = self._select_balanced_choice(scenarios, "scenario_id")
                rel = random.choice(scenario.relationships)
                inst = INSTITUTIONS.get(rel.institution_id)
                aud_id = random.choice(rel.audience_ids)
                aud = AUDIENCES.get(aud_id)
                act = next((a for a in scenario.actions if a.id in rel.action_ids), None)
                haz = next((h for h in scenario.hazards if h.id in rel.hazard_ids), None)
                
                if not all([inst, aud, act, haz]):
                    continue
                
                loc = random.choice(scenario.locations)
                
                # Balancing selector for other layers
                intent = self._select_balanced_choice(list(TEMPLATE_FAMILIES.keys()), "intent")
                severity = self._select_balanced_choice(list(TEMPLATE_FAMILIES[intent].keys()), "severity")
                pattern = self._select_balanced_choice(list(TEMPLATE_FAMILIES[intent][severity].keys()), "syntactic_pattern")
                tone = self._select_balanced_choice(["Informational", "Urgent", "Authoritative", "Community-outreach"], "tone")
                channel = self._select_balanced_choice(["Radio", "SMS", "Poster", "Social Media"], "distribution_channel")
                lexical_profile = "Emergency" if severity == "Emergency" else ("Formal" if intent == "Warning" else "Community outreach")
                
                scenario_config = {
                    "domain": domain,
                    "topic": scenario.topic,
                    "subtopic": scenario.subtopic,
                    "scenario_id": scenario.id,
                    "institution": inst.name,
                    "audience": aud.name,
                    "audience_name": aud.name,
                    "hazard": haz.name,
                    "location": loc.name,
                    "intent": intent,
                    "severity": severity,
                    "syntactic_pattern": pattern,
                    "lexical_profile": lexical_profile,
                    "tone": tone,
                    "distribution_channel": channel
                }
                
                batch_to_request = min(batch_size, self.target_per_domain - count)
                tasks.append((scenario_config, batch_to_request, domain_prefix))
                count += batch_to_request

        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = 10
        print(f"Starting parallel LLM generation of {self.size} PSAs with {max_workers} concurrent workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.llm_generator.generate_batch, config, req_size): (config, prefix)
                for config, req_size, prefix in tasks
            }
            
            completed_count = 0
            domain_counters = {d: 0 for d in DOMAINS}
            
            for future in as_completed(futures):
                config, prefix = futures[future]
                try:
                    batch_records = future.result()
                    for record in batch_records:
                        domain = config["domain"]
                        self.stats["domain"][domain] += 1
                        self.stats["scenario_id"][config["scenario_id"]] += 1
                        self.stats["intent"][config["intent"]] += 1
                        self.stats["severity"][config["severity"]] += 1
                        self.stats["syntactic_pattern"][config["syntactic_pattern"]] += 1
                        self.stats["tone"][config["tone"]] += 1
                        self.stats["distribution_channel"][config["distribution_channel"]] += 1
                        
                        domain_counters[domain] += 1
                        domain_count = self.start_counts.get(domain, 0) + domain_counters[domain]
                        psa_id = f"PSA_{prefix}_{domain_count:05d}"
                        english_records.append({
                            "PSA_Id": psa_id,
                            "Domain": domain,
                            "Topic": config["topic"],
                            "Subtopic": config["subtopic"],
                            "Class": "PSA",
                            "English": record["English"],
                            "Kiswahili": "",
                            "Somali": "",
                            "Luo": "",
                            "is_synthetic": True,
                            "model_version": "NLLB-200",
                            "scenario_id": config["scenario_id"],
                            "intent": record["intent"],
                            "severity": record["severity"],
                            "syntactic_pattern": record["syntactic_pattern"],
                            "lexical_profile": record["lexical_profile"],
                            "word_count": record["word_count"]
                        })
                        completed_count += 1
                        if completed_count % 100 == 0:
                            print(f"Progress: Generated and validated {completed_count} / {self.size} PSAs...")
                except Exception as e:
                    print(f"Batch generation error: {e}")
                    
        print(f"Parallel LLM generation complete! Total generated: {len(english_records)}")
        return english_records

