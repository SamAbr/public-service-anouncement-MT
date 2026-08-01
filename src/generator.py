import os
import json
import random
from collections import defaultdict
from .config import DOMAINS, MIN_WORDS, MAX_WORDS, CHECKPOINT_FILE
from .validator import ValidationEngine
from .translator import NLLBTranslator

# Import scenarios and entities
from .knowledge.scenarios import SCENARIOS, INSTITUTIONS, AUDIENCES, HAZARDS, LOCATIONS
from .knowledge.entities import Context
from .templates.families import TEMPLATE_FAMILIES

class PSAGenerator:
    def __init__(self, size=50000, translator=None, checkpoint_file=None, engine="azure_llm",
                 azure_api_key=None, azure_endpoint=None, azure_deployment=None, start_counts=None):
        self.size = size
        self.target_per_domain = size // len(DOMAINS)
        self.validator = ValidationEngine(min_words=10, max_words=25)
        self.translator = translator if translator else NLLBTranslator()
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

    def generate_and_translate(self):
        """
        Coordinates the full pipeline:
        1. Generate English PSAs.
        2. Resume from checkpoint if it exists.
        3. Translate Swahili, Somali, and Luo sequentially.
        4. Save/checkpoint progress.
        """
        records = self.generate_english_psas()
        total_to_translate = len(records)
        print(f"Total English PSAs generated: {total_to_translate}")
        
        checkpoint_states = {"Kiswahili": 0, "Somali": 0, "Luo": 0}
        
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    records = checkpoint.get("records", records)
                    checkpoint_states = checkpoint.get("checkpoint_states", checkpoint_states)
                    print(f"Resuming translation. Current indices: {checkpoint_states}")
            except Exception as e:
                print(f"Could not read checkpoint file: {e}. Starting fresh.")
                
        targets = [
            ("Kiswahili", "swh_Latn", "facebook/nllb-200-distilled-600M"),
            ("Somali", "som_Latn", "facebook/nllb-200-1.3B"),
            ("Luo", "luo_Latn", "facebook/nllb-200-1.3B")
        ]
        
        for col_name, lang_code, model_name in targets:
            start_idx = checkpoint_states.get(col_name, 0)
            if start_idx >= total_to_translate:
                print(f"Skipping {col_name}: Fully translated.")
                continue
                
            print(f"\n=== Translating to {col_name} using model {model_name} (from index {start_idx}) ===")
            
            self.translator.load_model(model_name=model_name)
            batch_size = self.translator.batch_size
            
            english_texts = [r["English"] for r in records]
            
            for i in range(start_idx, total_to_translate, batch_size):
                end_idx = min(i + batch_size, total_to_translate)
                batch_texts = english_texts[i:end_idx]
                
                batch_translations = self.translator.translate_batch(batch_texts, tgt_lang=lang_code)
                
                for idx, translation in enumerate(batch_translations):
                    record_idx = i + idx
                    records[record_idx][col_name] = translation
                    
                checkpoint_states[col_name] = end_idx
                checkpoint_data = {
                    "records": records,
                    "checkpoint_states": checkpoint_states
                }
                
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                    
                print(f"Translated {col_name} and checkpointed up to index {end_idx}/{total_to_translate}...")
                
            self.translator.unload_model()
            
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            
        return records
