import hashlib
import re

class Deduplicator:
    def __init__(self):
        # Store hashes of normalized English texts
        self.seen_texts = set()
        # Store core combinations (institution, action, hazard) to avoid near-duplicates
        self.seen_slots = set()

    def _normalize(self, text):
        """Normalize text by converting to lowercase and removing punctuation/whitespace."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return " ".join(text.split())

    def is_duplicate(self, text, slot_combination=None):
        """
        Checks if the text or the slot combination is a duplicate.
        """
        normalized = self._normalize(text)
        text_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
        
        # Check text hash
        if text_hash in self.seen_texts:
            return True
            
        # Check core slot combination (e.g. same institution issuing same action for same hazard)
        if slot_combination:
            slot_hash = hashlib.md5(str(slot_combination).encode('utf-8')).hexdigest()
            if slot_hash in self.seen_slots:
                return True
                
        return False

    def add(self, text, slot_combination=None):
        """
        Adds the text and slot combination to the deduplication registry.
        """
        normalized = self._normalize(text)
        text_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
        self.seen_texts.add(text_hash)
        
        if slot_combination:
            slot_hash = hashlib.md5(str(slot_combination).encode('utf-8')).hexdigest()
            self.seen_slots.add(slot_hash)
            
    def size(self):
        return len(self.seen_texts)
