import os
import json
import random
from config import DOMAINS, MIN_WORDS, MAX_WORDS, CHECKPOINT_FILE
from grammar import ControlledGrammarEngine
from deduplicator import Deduplicator
from validator import ValidationEngine
from translator import NLLBTranslator

# Import templates
from templates import education, agriculture, governance, health, security

# Import knowledge base
from knowledge import institutions, audiences, actions, hazards, locations, terminology

class PSAGenerator:
    def __init__(self, size=50000, translator=None):
        self.size = size
        self.target_per_domain = size // len(DOMAINS)
        self.grammar_engine = ControlledGrammarEngine()
        self.deduplicator = Deduplicator()
        self.validator = ValidationEngine(min_words=MIN_WORDS, max_words=MAX_WORDS)
        self.translator = translator if translator else NLLBTranslator()
        
        # Mapping domains to their templates
        self.templates_map = {
            "Education": education,
            "Agriculture": agriculture,
            "Governance": governance,
            "Health": health,
            "Security & Safety": security
        }

    def generate_english_psas(self):
        """
        Generates the target number of unique, valid English PSAs for all domains.
        Returns a list of dicts containing the English text and domain metadata.
        """
        english_records = []
        
        for domain in DOMAINS:
            print(f"Generating English PSAs for domain: '{domain}'...")
            domain_templates = self.templates_map[domain]
            
            count = 0
            attempts = 0
            max_attempts = self.target_per_domain * 20  # Safeguard against infinite loops
            
            # Fetch knowledge lists
            insts = institutions.get_institutions_for_domain(domain)
            auds = audiences.get_audiences_for_domain(domain)
            acts = actions.get_actions_for_domain(domain)
            hazs = hazards.get_hazards_for_domain(domain)
            locs = locations.get_locations_for_domain(domain)
            terms = terminology.get_terminology_for_domain(domain)
            
            while count < self.target_per_domain and attempts < max_attempts:
                attempts += 1
                
                # Pick a random template index to include in slots for deduplication
                template_idx = random.randint(0, len(domain_templates.TEMPLATES) - 1)
                
                # We also track slot combinations to prevent near-duplicates
                inst = random.choice(insts)
                aud = random.choice(auds)
                act = random.choice(acts)
                haz = random.choice(hazs)
                loc = random.choice(locs)
                
                # Follow up (70% probability)
                has_follow_up = random.random() < 0.70
                follow_up_idx = random.randint(0, len(domain_templates.FOLLOW_UPS) - 1) if has_follow_up else -1
                
                slot_combination = (domain, template_idx, inst, aud, act, haz, loc, follow_up_idx)
                
                # Generate sentence
                english_text = self.grammar_engine.generate_psa(
                    templates=domain_templates.TEMPLATES,
                    openings=domain_templates.OPENINGS,
                    follow_ups=domain_templates.FOLLOW_UPS,
                    institutions=insts,
                    audiences=auds,
                    actions=acts,
                    hazards=hazs,
                    locations=locs,
                    terminologies=terms
                )
                
                # Validate English
                is_valid, reason = self.validator.validate(english_text)
                if not is_valid:
                    continue
                    
                # Deduplicate English
                if self.deduplicator.is_duplicate(english_text, slot_combination):
                    continue
                    
                # Register
                self.deduplicator.add(english_text, slot_combination)
                
                # Unique ID format: PSA_EDU_00001
                domain_prefix = domain.split()[0][:3].upper()
                psa_id = f"PSA_{domain_prefix}_{count+1:05d}"
                
                english_records.append({
                    "PSA_Id": psa_id,
                    "Domain": domain,
                    "Class": "PSA",
                    "English": english_text
                })
                count += 1
                
            print(f"Generated {count} English records for '{domain}' after {attempts} attempts.")
            
        return english_records

    def generate_and_translate(self):
        """
        Coordinates the full pipeline:
        1. Generate English PSAs.
        2. Resume from checkpoint if it exists.
        3. Translate in batches with NLLB-200.
        4. Save/checkpoint progress.
        """
        # Step 1: Generate or load English PSAs
        english_records = self.generate_english_psas()
        total_to_translate = len(english_records)
        print(f"Total English PSAs generated: {total_to_translate}")
        
        # Step 2: Check for checkpoint
        translated_records = []
        start_idx = 0
        
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    translated_records = checkpoint.get("records", [])
                    start_idx = checkpoint.get("next_index", 0)
                    print(f"Resuming from checkpoint. Already translated {start_idx}/{total_to_translate} records.")
            except Exception as e:
                print(f"Could not read checkpoint file: {e}. Starting fresh.")
                
        # If the generated size is different or we start fresh, reset
        if start_idx >= total_to_translate:
            print("Checkpoint matches or exceeds generated size. Completed.")
            return translated_records
            
        # Step 3: Run translation
        print("Initializing translator...")
        self.translator.load_model()
        
        batch_size = self.translator.batch_size
        english_texts = [r["English"] for r in english_records]
        
        # Translate remaining in batches
        for i in range(start_idx, total_to_translate, batch_size):
            end_idx = min(i + batch_size, total_to_translate)
            batch_texts = english_texts[i:end_idx]
            
            # Translate batch
            batch_translations = self.translator.translate_batch(batch_texts)
            
            # Save translated records
            for idx, translation in enumerate(batch_translations):
                record_idx = i + idx
                record = english_records[record_idx].copy()
                record["Kiswahili"] = translation
                translated_records.append(record)
                
            # Write checkpoint
            checkpoint_data = {
                "records": translated_records,
                "next_index": end_idx
            }
            with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                
            print(f"Translated and checkpointed up to index {end_idx}/{total_to_translate}...")
            
        # Remove checkpoint file on successful completion
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            
        return translated_records
