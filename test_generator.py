import unittest
import os
import shutil
import tempfile
from src.grammar import ControlledGrammarEngine
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
        import random
        random.seed(0)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_grammar_engine(self):
        engine = ControlledGrammarEngine()
        template = "{opening} {institution} {audience} {action_infinitive} {hazard} {location}."
        opening = "Public Alert:"
        institution = "Ministry of Health"
        audience = "residents"
        action_infinitive = "[boil] drinking water"
        action_imperative = "[boil] drinking water"
        action_noun = "[boil] drinking water"
        hazard = "cholera outbreak"
        location = "in cities"
        
        psa = engine.generate_psa(
            template=template,
            opening=opening,
            institution=institution,
            audience=audience,
            action_infinitive=action_infinitive,
            action_imperative=action_imperative,
            action_noun=action_noun,
            hazard=hazard,
            location=location
        )
        self.assertIn("Public Alert: Ministry of Health residents", psa)
        self.assertNotIn("[boil]", psa)  # Verify bracketed verb synonymized

    def test_validation_engine(self):
        validator = ValidationEngine(min_words=10, max_words=25)
        
        # Test valid sentence (length within limits, capitalization, ends in punctuation, contains PSA keywords)
        valid_text = "The Ministry of Health advises all citizens to boil drinking water today in sub-counties."
        is_valid, reason = validator.validate(valid_text)
        self.assertTrue(is_valid, f"Failed validation: {reason}")
        
        # Test short text (10-25, "Wash hands." is too short)
        short_text = "Wash hands."
        is_valid, reason = validator.validate(short_text)
        self.assertFalse(is_valid)
        
        # Test missing punctuation
        no_punc = "The Ministry of Health advises all citizens to boil drinking water today"
        is_valid, reason = validator.validate(no_punc)
        self.assertFalse(is_valid)

        # Test double modifiers
        duplicate_mod = "The Ministry urges all citizens to boil water immediately and immediately go to clinic."
        is_valid, reason = validator.validate(duplicate_mod)
        self.assertFalse(is_valid)

        # Test Press Release rejection
        press_release = "The Cabinet Secretary launched a new national vaccination drive at the Ministry headquarters."
        is_valid, reason = validator.validate(press_release)
        self.assertFalse(is_valid)

        # Test Gazette Notice rejection
        gazette_notice = "Pursuant to section 5, the general public is notified of registration guidelines."
        is_valid, reason = validator.validate(gazette_notice)
        self.assertFalse(is_valid)

        # Test multiple sentences behavior (allows first one up to 10% budget, then rejects subsequent ones)
        multi_sentence_1 = "The Ministry urges all citizens to boil drinking water today. Clean water is safe for health."
        is_valid_1, reason_1 = validator.validate(multi_sentence_1)
        self.assertTrue(is_valid_1)  # First outlier is allowed

        multi_sentence_2 = "Avoid crossing flooded rivers during rainy seasons. Follow local emergency advice immediately."
        is_valid_2, reason_2 = validator.validate(multi_sentence_2)
        self.assertFalse(is_valid_2)  # Second consecutive outlier is rejected (ratio 50% > 10%)

        # Test three sentences (always rejected)
        three_sentence = "Boil all drinking water today. Clean water is safe. Protect your family."
        is_valid_3, reason_3 = validator.validate(three_sentence)
        self.assertFalse(is_valid_3)

    def test_full_pipeline_mocked(self):
        # Generate 25 parallel records (5 per domain)
        test_checkpoint = os.path.join(self.temp_dir, "test_checkpoint.json")
        generator = PSAGenerator(size=25, translator=self.mock_translator, checkpoint_file=test_checkpoint)
        records = generator.generate_and_translate()
        
        self.assertEqual(len(records), 25)
        
        # Verify metadata structure
        for record in records:
            self.assertIn("PSA_Id", record)
            self.assertIn("Domain", record)
            self.assertIn("Topic", record)
            self.assertIn("Subtopic", record)
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
            self.assertIn("scenario_id", record)
            self.assertIn("intent", record)
            self.assertIn("severity", record)
            self.assertIn("syntactic_pattern", record)
            self.assertIn("lexical_profile", record)
            self.assertIn("word_count", record)
            
        # Export
        exporter = Exporter(self.output_csv)
        exporter.export(records)
        
        self.assertTrue(os.path.exists(self.output_csv))
        
        # Read file to check lines count (25 records + 1 header = 26 lines)
        with open(self.output_csv, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 26)

if __name__ == "__main__":
    unittest.main()
