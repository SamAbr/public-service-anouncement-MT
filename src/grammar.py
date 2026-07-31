import random

class ControlledGrammarEngine:
    def __init__(self):
        # We will import templates dynamically or pass them during generation
        pass

    def generate_psa(self, templates, openings, follow_ups, institutions, audiences, actions, hazards, locations, terminologies):
        """
        Generates a structured PSA sentence using the provided components.
        """
        # Choose random components
        opening = random.choice(openings)
        institution = random.choice(institutions)
        audience = random.choice(audiences)
        action = random.choice(actions)
        hazard = random.choice(hazards)
        location = random.choice(locations)
        
        # Choose a template
        template = random.choice(templates)
        
        # Format the main PSA sentence
        main_sentence = template.format(
            opening=opening,
            institution=institution,
            audience=audience,
            action=action,
            hazard=hazard,
            location=location
        )
        
        # Decided whether to include a follow-up sentence
        # 70% chance of adding a follow-up to increase length and variation
        if random.random() < 0.70 and follow_ups:
            follow_up = random.choice(follow_ups)
            # We can optionally inject a terminology keyword into the follow-up
            if "{term}" in follow_up and terminologies:
                term = random.choice(terminologies)
                follow_up = follow_up.format(term=term)
            
            # Combine sentences
            psa = f"{main_sentence} {follow_up}"
        else:
            psa = main_sentence

        # Apply a micro-paraphrasing pass to vary syntax/vocabulary and avoid template rigidity
        substitutions = {
            " advises ": [" urges ", " directs ", " requests ", " calls upon ", " exhorts "],
            " immediately ": [" without delay ", " urgently ", " promptly "],
            " to mitigate the impact of ": [" to curb the threat of ", " to protect against ", " to reduce the risks of "],
            " in response to ": [" following ", " in light of ", " due to rising cases of "],
            " online via the official web portal": [" via the official portal", " online through the official portal", " through the portal online"]
        }
        for target, choices in substitutions.items():
            if target in psa and random.random() < 0.5:
                psa = psa.replace(target, random.choice(choices), 1)
            
        # Clean up double spaces, strip whitespace
        psa = " ".join(psa.split())
        return psa
