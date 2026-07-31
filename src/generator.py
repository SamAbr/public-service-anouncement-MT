import os
import json
import random
from .config import DOMAINS, MIN_WORDS, MAX_WORDS, CHECKPOINT_FILE
from .grammar import ControlledGrammarEngine
from .deduplicator import Deduplicator
from .validator import ValidationEngine
from .translator import NLLBTranslator

# Import templates
from .templates import education, agriculture, governance, health, security

# Import knowledge base
from .knowledge import institutions, audiences, actions, hazards, locations, terminology

class PSAGenerator:
    def __init__(self, size=50000, translator=None, checkpoint_file=None):
        self.size = size
        self.target_per_domain = size // len(DOMAINS)
        self.grammar_engine = ControlledGrammarEngine()
        self.deduplicator = Deduplicator()
        self.validator = ValidationEngine(min_words=MIN_WORDS, max_words=MAX_WORDS)
        self.translator = translator if translator else NLLBTranslator()
        self.checkpoint_file = checkpoint_file if checkpoint_file else CHECKPOINT_FILE
        
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
                
                # 1. Pick a random template index
                template_idx = random.randint(0, len(domain_templates.TEMPLATES) - 1)
                
                # 2. Pick a thematic tag from the elements in the domain lists (excluding "general" if possible)
                all_tags = set()
                for items in [insts, acts, hazs]:
                    for item in items:
                        if isinstance(item, tuple) and len(item) > 1:
                            all_tags.update(item[1])
                specific_tags = all_tags - {"general"}
                chosen_tag = random.choice(list(specific_tags)) if specific_tags else (random.choice(list(all_tags)) if all_tags else None)

                # Helper to filter items by the selected tag
                def filter_by_tag(item_list, tag):
                    if not tag:
                        return [item[0] if isinstance(item, tuple) else item for item in item_list]
                    filtered = []
                    for item in item_list:
                        if isinstance(item, tuple):
                            if tag in item[1] or "general" in item[1]:
                                filtered.append(item[0])
                        else:
                            filtered.append(item)
                    return filtered if filtered else [item[0] if isinstance(item, tuple) else item for item in item_list]

                filtered_insts = filter_by_tag(insts, chosen_tag)
                filtered_acts = filter_by_tag(acts, chosen_tag)
                filtered_hazs = filter_by_tag(hazs, chosen_tag)

                # 3. Select coherent slots
                inst = random.choice(filtered_insts)
                aud = random.choice(auds)
                act = random.choice(filtered_acts)
                haz = random.choice(filtered_hazs)
                loc = random.choice(locs)
                
                # Follow up (70% probability)
                has_follow_up = random.random() < 0.70
                follow_up_idx = random.randint(0, len(domain_templates.FOLLOW_UPS) - 1) if has_follow_up else -1
                
                slot_combination = (domain, template_idx, inst, aud, act, haz, loc, follow_up_idx)
                
                # Generate sentence (we pass single-item lists to guarantee slot_combination matches the generated text)
                english_text = self.grammar_engine.generate_psa(
                    templates=domain_templates.TEMPLATES,
                    openings=domain_templates.OPENINGS,
                    follow_ups=domain_templates.FOLLOW_UPS,
                    institutions=[inst],
                    audiences=[aud],
                    actions=[act],
                    hazards=[haz],
                    locations=[loc],
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
                    "English": english_text,
                    "Kiswahili": "",
                    "Somali": "",
                    "Luo": "",
                    "is_synthetic": True,
                    "model_version": "NLLB-200",
                    "template_id": f"T_{domain_prefix}_{template_idx}"
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
