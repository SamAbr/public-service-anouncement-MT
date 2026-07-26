import csv
import random
import os

class Exporter:
    def __init__(self, output_file):
        self.output_file = output_file

    def export(self, records):
        """
        Exports the records to the output CSV file.
        Each record should be a dict: {
            "PSA_Id": str,
            "Domain": str,
            "Class": "PSA",
            "English": str,
            "Kiswahili": str
        }
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

        fieldnames = ["PSA_Id", "Domain", "Class", "English", "Kiswahili"]

        print(f"Writing {len(shuffled_records)} interleaved records to {self.output_file}...")
        with open(self.output_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in shuffled_records:
                writer.writerow({
                    "PSA_Id": record.get("PSA_Id"),
                    "Domain": record.get("Domain"),
                    "Class": record.get("Class", "PSA"),
                    "English": record.get("English"),
                    "Kiswahili": record.get("Kiswahili")
                })
        print("Export completed successfully.")
