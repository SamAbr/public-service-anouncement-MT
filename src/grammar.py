import random
import re

SYNONYMS = {
    "verify": ["verify", "check", "confirm", "validate", "review"],
    "adhere": ["adhere to", "comply with", "follow", "abide by", "observe"],
    "report": ["report", "notify", "inform", "alert", "lodge a report on"],
    "submit": ["submit", "file", "present", "forward", "lodge"],
    "update": ["update", "refresh", "amend", "modify"],
    "apply": ["apply", "use", "deploy", "utilize"],
    "adopt": ["adopt", "implement", "utilize", "embrace"],
    "stockpile": ["stockpile", "store", "accumulate", "reserve", "lay in"],
    "avoid": ["avoid", "steer clear of", "shun", "evade"],
    "relocate": ["relocate", "move", "evacuate", "shift"],
    "boil": ["boil", "purify", "treat"],
    "register": ["register for", "enroll in", "sign up for"]
}

class ControlledGrammarEngine:
    def __init__(self):
        pass

    def resolve_lexical_variants(self, text: str) -> str:
        """
        Dynamically replaces bracketed verbs (e.g., [verify]) with random synonyms,
        preserving uppercase capitalization.
        """
        def replacer(match):
            raw_verb = match.group(1)
            verb = raw_verb.lower()
            if verb in SYNONYMS:
                choice = random.choice(SYNONYMS[verb])
                if raw_verb and raw_verb[0].isupper():
                    choice = choice[0].upper() + choice[1:]
                return choice
            return raw_verb
        return re.sub(r'\[(.*?)\]', replacer, text)

    def generate_psa(self, template: str, opening: str, institution: str, audience: str, 
                     action_infinitive: str, action_imperative: str, action_noun: str,
                     hazard: str, location: str, terminology: str = "", season: str = "general") -> str:
        """
        Formats a template using scenario entities and processes lexical synonym substitution.
        """
        # Format placeholders
        formatted = template.format(
            opening=opening,
            institution=institution,
            audience=audience,
            action_infinitive=action_infinitive,
            action_imperative=action_imperative,
            action_noun=action_noun,
            hazard=hazard,
            location=location,
            terminology=terminology,
            season=season
        )
        
        # Apply lexical variant substitutions
        psa = self.resolve_lexical_variants(formatted)
        
        # Clean double spaces and strip
        psa = " ".join(psa.split())
        
        # Ensure correct sentence ending punctuation
        if not psa.endswith((".", "!", "?")):
            psa += "."
            
        return psa
