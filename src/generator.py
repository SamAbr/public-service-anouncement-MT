import os
import json
import random
from collections import defaultdict
from .config import DOMAINS, MIN_WORDS, MAX_WORDS, CHECKPOINT_FILE
from .grammar import ControlledGrammarEngine
from .validator import ValidationEngine
from .translator import NLLBTranslator

# Import new scenarios and entities
from .knowledge.scenarios import SCENARIOS, INSTITUTIONS, AUDIENCES, HAZARDS, LOCATIONS
from .knowledge.entities import Context
from .templates.families import TEMPLATE_FAMILIES, OPENINGS

class PSAGenerator:
    def __init__(self, size=50000, translator=None, checkpoint_file=None):
        self.size = size
        self.target_per_domain = size // len(DOMAINS)
        self.grammar_engine = ControlledGrammarEngine()
        self.validator = ValidationEngine(min_words=12, max_words=40)
        self.translator = translator if translator else NLLBTranslator()
        self.checkpoint_file = checkpoint_file if checkpoint_file else CHECKPOINT_FILE
        
        # Corpus-Level Balancing Controller state
        self.stats = {
            "domain": defaultdict(int),
            "scenario_id": defaultdict(int),
            "intent": defaultdict(int),
            "severity": defaultdict(int),
            "syntactic_pattern": defaultdict(int),
            "template_use": defaultdict(int)
        }

    def _select_balanced_choice(self, choices, stat_category):
        """
        Implements the Corpus-Level Balancing Controller:
        Selects a choice that has the minimum frequency generated so far to ensure even distribution.
        """
        if not choices:
            return None
        # Sort choices by their occurrence stats (ascending)
        scored_choices = [(choice, self.stats[stat_category][str(choice)]) for choice in choices]
        # Find min frequency
        min_freq = min(score for _, score in scored_choices)
        # Select randomly from all candidates that share the minimum frequency
        candidates = [choice for choice, score in scored_choices if score == min_freq]
        return random.choice(candidates)

    def generate_english_psas(self):
        """
        Generates the target number of unique, valid, and semantically diverse English PSAs.
        Returns a list of dicts containing the English text and domain metadata.
        """
        english_records = []
        
        for domain in DOMAINS:
            print(f"Generating English PSAs for domain: '{domain}'...")
            scenarios = SCENARIOS.get(domain, [])
            if not scenarios:
                continue
                
            count = 0
            attempts = 0
            max_attempts = self.target_per_domain * 20  # Safeguard against infinite loops
            
            while count < self.target_per_domain and attempts < max_attempts:
                attempts += 1
                
                # 1. Select Scenario (Balancing Controller)
                scenario = self._select_balanced_choice(scenarios, "scenario_id")
                
                # 2. Select Relationship Constraint (logical grouping of inst, aud, act, haz)
                rel = random.choice(scenario.relationships)
                
                inst = INSTITUTIONS.get(rel.institution_id)
                aud_id = random.choice(rel.audience_ids)
                aud = AUDIENCES.get(aud_id)
                act = next((a for a in scenario.actions if a.id in rel.action_ids), None)
                haz = next((h for h in scenario.hazards if h.id in rel.hazard_ids), None)
                
                if not all([inst, aud, act, haz]):
                    continue
                    
                # Select random location and terminology from scenario pools
                loc = random.choice(scenario.locations)
                term = random.choice(scenario.terminology) if scenario.terminology else ""
                
                # 3. Context Builder
                season = "general"
                if scenario.allowed_seasons != ["any"]:
                    season = random.choice(scenario.allowed_seasons)
                
                context = Context(
                    season=season,
                    weather="heavy rainfall" if season == "rainy" else ("dry spell" if season == "dry" else "normal weather"),
                    school_calendar="exam period" if scenario.id == "exam_security" else "normal term"
                )
                
                # 4. Select Intent & Severity (Balancing Controller)
                # Filter intents available in TEMPLATE_FAMILIES
                intents = list(TEMPLATE_FAMILIES.keys())
                intent = self._select_balanced_choice(intents, "intent")
                
                severities = list(TEMPLATE_FAMILIES[intent].keys())
                severity = self._select_balanced_choice(severities, "severity")
                
                patterns = list(TEMPLATE_FAMILIES[intent][severity].keys())
                pattern = self._select_balanced_choice(patterns, "syntactic_pattern")
                
                # 5. Select template
                templates_list = TEMPLATE_FAMILIES[intent][severity][pattern]
                template = self._select_balanced_choice(templates_list, "template_use")
                
                opening = random.choice(OPENINGS)
                
                # 6. Realize PSA sentence via grammar engine
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
                
                # 7. Realism Validation
                is_valid, reason = self.validator.validate(english_text)
                if not is_valid:
                    continue
                    
                # 8. Semantic Similarity Check (Cosine Similarity < 0.92 / Jaccard)
                if not self.validator.is_semantically_unique(english_text, threshold=0.92):
                    continue
                    
                # 9. Register statistics on success
                self.stats["domain"][domain] += 1
                self.stats["scenario_id"][scenario.id] += 1
                self.stats["intent"][intent] += 1
                self.stats["severity"][severity] += 1
                self.stats["syntactic_pattern"][pattern] += 1
                self.stats["template_use"][template] += 1
                
                # Unique ID format: PSA_EDU_00001
                domain_prefix = domain.split()[0][:3].upper()
                psa_id = f"PSA_{domain_prefix}_{count+1:05d}"
                
                # Lexical profile mapping
                lexical_profile = "Emergency" if severity == "Emergency" else ("Formal" if intent == "Warning" else "Community outreach")
                
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
        # Step 1: Generate or load English PSAs
        records = self.generate_english_psas()
        total_to_translate = len(records)
        print(f"Total English PSAs generated: {total_to_translate}")
        
        # Step 2: Check for checkpoint
        checkpoint_states = {
            "Kiswahili": 0,
            "Somali": 0,
            "Luo": 0
        }
        
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    records = checkpoint.get("records", records)
                    checkpoint_states = checkpoint.get("checkpoint_states", checkpoint_states)
                    print(f"Resuming translation. Current indices: {checkpoint_states}")
            except Exception as e:
                print(f"Could not read checkpoint file: {e}. Starting fresh.")
                
        # Define NLLB model mapping per target language
        targets = [
            ("Kiswahili", "swh_Latn", "facebook/nllb-200-distilled-600M"),
            ("Somali", "som_Latn", "facebook/nllb-200-1.3B"),
            ("Luo", "luo_Latn", "facebook/nllb-200-1.3B")
        ]
        
        # Step 3: Run translation sequentially per language
        for col_name, lang_code, model_name in targets:
            start_idx = checkpoint_states.get(col_name, 0)
            if start_idx >= total_to_translate:
                print(f"Skipping {col_name}: Fully translated.")
                continue
                
            print(f"\n=== Translating to {col_name} using model {model_name} (from index {start_idx}) ===")
            
            # Load translation model
            self.translator.load_model(model_name=model_name)
            batch_size = self.translator.batch_size
            
            english_texts = [r["English"] for r in records]
            
            for i in range(start_idx, total_to_translate, batch_size):
                end_idx = min(i + batch_size, total_to_translate)
                batch_texts = english_texts[i:end_idx]
                
                # Translate batch
                batch_translations = self.translator.translate_batch(batch_texts, tgt_lang=lang_code)
                
                # Update records in place
                for idx, translation in enumerate(batch_translations):
                    record_idx = i + idx
                    records[record_idx][col_name] = translation
                    
                # Update checkpoint state
                checkpoint_states[col_name] = end_idx
                checkpoint_data = {
                    "records": records,
                    "checkpoint_states": checkpoint_states
                }
                
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                    
                print(f"Translated {col_name} and checkpointed up to index {end_idx}/{total_to_translate}...")
                
            # Unload model to release GPU memory before loading the next one
            self.translator.unload_model()
            
        # Remove checkpoint file on successful completion of all languages
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            
        return records
