import os
import json
import random
from collections import defaultdict
from .config import DOMAINS, MIN_WORDS, MAX_WORDS, CHECKPOINT_FILE
from .grammar import ControlledGrammarEngine
from .validator import ValidationEngine
from .translator import NLLBTranslator

# Import scenarios, entities, and templates
from .knowledge.scenarios import SCENARIOS, INSTITUTIONS, AUDIENCES, HAZARDS, LOCATIONS
from .knowledge.entities import Context
from .templates.families import TEMPLATE_FAMILIES, DOMAIN_OPENINGS

class PSAGenerator:
    def __init__(self, size=50000, translator=None, checkpoint_file=None, engine="templates",
                 azure_api_key=None, azure_endpoint=None, azure_deployment=None):
        self.size = size
        self.target_per_domain = size // len(DOMAINS)
        self.grammar_engine = ControlledGrammarEngine()
        self.validator = ValidationEngine(min_words=10, max_words=25)
        self.translator = translator if translator else NLLBTranslator()
        self.checkpoint_file = checkpoint_file if checkpoint_file else CHECKPOINT_FILE
        self.engine = engine
        
        self.llm_generator = None
        if self.engine == "azure_llm":
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
            "template_use": defaultdict(int),
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
        Generates the target number of unique, valid English PSAs.
        Supports both template-based grammar engine and Azure OpenAI (GPT-4o) backend.
        """
        english_records = []
        
        for domain in DOMAINS:
            print(f"Generating English PSAs for domain: '{domain}'...")
            scenarios = SCENARIOS.get(domain, [])
            if not scenarios:
                continue
                
            count = 0
            attempts = 0
            max_attempts = self.target_per_domain * 25  # Safeguard limits
            
            while count < self.target_per_domain and attempts < max_attempts:
                attempts += 1
                
                # 1. Select Scenario (Balancing Controller)
                scenario = self._select_balanced_choice(scenarios, "scenario_id")
                
                # 2. Select Relationship Constraint
                rel = random.choice(scenario.relationships)
                inst = INSTITUTIONS.get(rel.institution_id)
                aud_id = random.choice(rel.audience_ids)
                aud = AUDIENCES.get(aud_id)
                act = next((a for a in scenario.actions if a.id in rel.action_ids), None)
                haz = next((h for h in scenario.hazards if h.id in rel.hazard_ids), None)
                
                if not all([inst, aud, act, haz]):
                    continue
                    
                loc = random.choice(scenario.locations)
                term = random.choice(scenario.terminology) if scenario.terminology else ""
                
                # Select context variables
                season = "general"
                if scenario.allowed_seasons != ["any"]:
                    season = random.choice(scenario.allowed_seasons)
                
                context = Context(
                    season=season,
                    weather="heavy rainfall" if season == "rainy" else ("dry spell" if season == "dry" else "normal weather"),
                    school_calendar="exam period" if scenario.id == "exam_security" else "normal term"
                )
                
                # Balancing selector for other layers
                intent = self._select_balanced_choice(list(TEMPLATE_FAMILIES.keys()), "intent")
                severity = self._select_balanced_choice(list(TEMPLATE_FAMILIES[intent].keys()), "severity")
                pattern = self._select_balanced_choice(list(TEMPLATE_FAMILIES[intent][severity].keys()), "syntactic_pattern")
                tone = self._select_balanced_choice(["Informational", "Urgent", "Authoritative", "Community-outreach"], "tone")
                channel = self._select_balanced_choice(["Radio", "SMS", "Poster", "Social Media"], "distribution_channel")
                
                lexical_profile = "Emergency" if severity == "Emergency" else ("Formal" if intent == "Warning" else "Community outreach")
                
                # Domain prefix for PSA IDs
                domain_prefix = domain.split()[0][:3].upper()
                
                if self.engine == "azure_llm" and self.llm_generator and self.llm_generator.is_configured():
                    # Batch generate from LLM
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
                    
                    batch_to_request = min(5, self.target_per_domain - count)
                    batch_records = self.llm_generator.generate_batch(scenario_config, batch_size=batch_to_request)
                    
                    for record in batch_records:
                        english_text = record["English"]
                        # Success: Register metrics & build record
                        self.stats["domain"][domain] += 1
                        self.stats["scenario_id"][scenario.id] += 1
                        self.stats["intent"][intent] += 1
                        self.stats["severity"][severity] += 1
                        self.stats["syntactic_pattern"][pattern] += 1
                        self.stats["tone"][tone] += 1
                        self.stats["distribution_channel"][channel] += 1
                        
                        psa_id = f"PSA_{domain_prefix}_{count+1:05d}"
                        english_records.append({
                            "PSA_Id": psa_id,
                            "Domain": domain,
                            "Topic": scenario.topic,
                            "Subtopic": scenario.subtopic,
                            "Class": "PSA",
                            "English": english_text,
                            "Kiswahili": "",
                            "Somali": "",
                            "Luo": "",
                            "is_synthetic": True,
                            "model_version": "NLLB-200",
                            "scenario_id": scenario.id,
                            "intent": record["intent"],
                            "severity": record["severity"],
                            "syntactic_pattern": record["syntactic_pattern"],
                            "lexical_profile": record["lexical_profile"],
                            "word_count": record["word_count"]
                        })
                        count += 1
                else:
                    # Template-based realization fallback
                    templates_list = TEMPLATE_FAMILIES[intent][severity][pattern]
                    template = self._select_balanced_choice(templates_list, "template_use")
                    openings = DOMAIN_OPENINGS.get(domain, ["Official Advisory:"])
                    opening = random.choice(openings)
                    
                    english_text = self.grammar_engine.generate_psa(
                        template=template,
                        opening=opening,
                        institution=inst.name,
                        audience=aud.name,
                        action_infinitive=act.infinitive,
                        action_imperative=act.imperative,
                        action_noun=act.noun,
                        hazard=haz.name,
                        location=loc.name,
                        terminology=term,
                        season=context.season
                    )
                    
                    # Grammar cleanup: remove duplicate "to to"
                    while "to to" in english_text:
                        english_text = english_text.replace("to to", "to")
                    while "  " in english_text:
                        english_text = english_text.replace("  ", " ")
                    
                    is_valid, reason = self.validator.validate(english_text)
                    if not is_valid:
                        continue
                        
                    if not self.validator.is_semantically_unique(english_text, threshold=0.92):
                        continue
                        
                    # Register stats
                    self.stats["domain"][domain] += 1
                    self.stats["scenario_id"][scenario.id] += 1
                    self.stats["intent"][intent] += 1
                    self.stats["severity"][severity] += 1
                    self.stats["syntactic_pattern"][pattern] += 1
                    self.stats["template_use"][template] += 1
                    self.stats["tone"][tone] += 1
                    self.stats["distribution_channel"][channel] += 1
                    
                    psa_id = f"PSA_{domain_prefix}_{count+1:05d}"
                    english_records.append({
                        "PSA_Id": psa_id,
                        "Domain": domain,
                        "Topic": scenario.topic,
                        "Subtopic": scenario.subtopic,
                        "Class": "PSA",
                        "English": english_text,
                        "Kiswahili": "",
                        "Somali": "",
                        "Luo": "",
                        "is_synthetic": True,
                        "model_version": "NLLB-200",
                        "scenario_id": scenario.id,
                        "intent": intent,
                        "severity": severity,
                        "syntactic_pattern": pattern,
                        "lexical_profile": lexical_profile,
                        "word_count": len(english_text.split())
                    })
                    count += 1
                    
            print(f"Generated {count} English records for '{domain}' after {attempts} attempts.")
            
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
