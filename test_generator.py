import unittest
import os
import shutil
import tempfile
from src.config import DOMAINS
from src.grammar import ControlledGrammarEngine
from src.deduplicator import Deduplicator
from src.validator import ValidationEngine
from src.exporter import Exporter
from src.generator import PSAGenerator

class MockTranslator:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def load_model(self, model_name=None):
        pass

    def unload_model(self):
        pass

    def translate_batch(self, texts, tgt_lang=None):
        suffix = " - Swahili Translation"
        if tgt_lang == "som_Latn":
            suffix = " - Somali Translation"
        elif tgt_lang == "luo_Latn":
            suffix = " - Luo Translation"
        return [f"{text}{suffix}" for text in texts]

class TestPSAGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_csv = os.path.join(self.temp_dir, "test_output.csv")
        self.mock_translator = MockTranslator()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_grammar_engine(self):
        engine = ControlledGrammarEngine()
        templates = ["{opening} {institution} {audience} {action} {hazard} {location}."]
        openings = ["Public Alert:"]
        follow_ups = ["Follow guidelines."]
        institutions = ["Min Health"]
        audiences = ["all residents"]
        actions = ["wash hands"]
        hazards = ["cholera outbreak"]
        locations = ["in cities"]
        
        psa = engine.generate_psa(
            templates=templates,
            openings=openings,
            follow_ups=follow_ups,
            institutions=institutions,
            audiences=audiences,
            actions=actions,
            hazards=hazards,
            locations=locations,
            terminologies=["hygiene"]
        )
        self.assertIn("Public Alert: Min Health all residents wash hands cholera outbreak in cities.", psa)

    def test_validation_engine(self):
        validator = ValidationEngine(min_words=5, max_words=15)
        
        # Test word range
        valid_text = "The Ministry of Health advises all citizens to boil drinking water today."
        is_valid, reason = validator.validate(valid_text)
        self.assertTrue(is_valid, f"Failed validation: {reason}")
        
        # Test short text
        short_text = "Wash hands."
        is_valid, reason = validator.validate(short_text)
        self.assertFalse(is_valid)
        
        # Test punctuation
        no_punc = "The Ministry of Health advises all citizens to boil drinking water"
        is_valid, reason = validator.validate(no_punc)
        self.assertFalse(is_valid)

    def test_deduplicator(self):
        dedup = Deduplicator()
        text = "This is a unique public announcement."
        slots = ("Education", 0, "Ministry", "Students", "Learn")
        
        self.assertFalse(dedup.is_duplicate(text, slots))
        dedup.add(text, slots)
        
        # Check duplicate
        self.assertTrue(dedup.is_duplicate(text, slots))
        self.assertTrue(dedup.is_duplicate("THIS IS A UNIQUE PUBLIC ANNOUNCEMENT."))

    def test_full_pipeline_mocked(self):
        # Generate a small dataset (e.g. 50 pairs, 10 per domain)
        test_checkpoint = os.path.join(self.temp_dir, "test_checkpoint.json")
        generator = PSAGenerator(size=50, translator=self.mock_translator, checkpoint_file=test_checkpoint)
        records = generator.generate_and_translate()
        
        self.assertEqual(len(records), 50)
        
        # Verify structure
        for record in records:
            self.assertIn("PSA_Id", record)
            self.assertIn("Domain", record)
            self.assertEqual(record["Class"], "PSA")
            self.assertIn("English", record)
            self.assertIn("Kiswahili", record)
            self.assertIn("Somali", record)
            self.assertIn("Luo", record)
            self.assertTrue(record["Kiswahili"].endswith("- Swahili Translation"))
            self.assertTrue(record["Somali"].endswith("- Somali Translation"))
            self.assertTrue(record["Luo"].endswith("- Luo Translation"))
            self.assertEqual(record["is_synthetic"], True)
            self.assertEqual(record["model_version"], "NLLB-200")
            self.assertTrue(record["template_id"].startswith("T_"))
            
        # Export
        exporter = Exporter(self.output_csv)
        exporter.export(records)
        
        self.assertTrue(os.path.exists(self.output_csv))
        
        # Read file to check lines count (50 records + 1 header = 51 lines)
        with open(self.output_csv, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 51)

if __name__ == "__main__":
    unittest.main()
