import csv
import random
import os

class Exporter:
    def __init__(self, output_file):
        self.output_file = output_file

    def export(self, records):
        """
        Exports the records to the output CSV file.
        The records are randomly shuffled (interleaved) before being written.
        """
        if not records:
            print("No records to export.")
            return

        # Interleave domain rows randomly
        shuffled_records = list(records)
        random.shuffle(shuffled_records)

        # Make sure parent directory exists
        parent_dir = os.path.dirname(os.path.abspath(self.output_file))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        fieldnames = [
            "PSA_Id", "Domain", "Topic", "Subtopic", "Class", "English",
            "Kiswahili", "Somali", "Luo", "is_synthetic", "model_version",
            "scenario_id", "intent", "severity", "syntactic_pattern",
            "lexical_profile", "word_count"
        ]

        print(f"Writing {len(shuffled_records)} interleaved records to {self.output_file}...")
        with open(self.output_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in shuffled_records:
                writer.writerow({
                    "PSA_Id": record.get("PSA_Id"),
                    "Domain": record.get("Domain"),
                    "Topic": record.get("Topic", ""),
                    "Subtopic": record.get("Subtopic", ""),
                    "Class": record.get("Class", "PSA"),
                    "English": record.get("English"),
                    "Kiswahili": record.get("Kiswahili", ""),
                    "Somali": record.get("Somali", ""),
                    "Luo": record.get("Luo", ""),
                    "is_synthetic": record.get("is_synthetic", True),
                    "model_version": record.get("model_version", "NLLB-200"),
                    "scenario_id": record.get("scenario_id", ""),
                    "intent": record.get("intent", ""),
                    "severity": record.get("severity", ""),
                    "syntactic_pattern": record.get("syntactic_pattern", ""),
                    "lexical_profile": record.get("lexical_profile", ""),
                    "word_count": record.get("word_count", 0)
                })
        print("Export completed successfully.")
