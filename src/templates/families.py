# Simplified Template Families mapped by (Intent, Severity, SyntacticPattern)
# Designed to remain strictly within the 10-25 word boundary for authentic PSAs.

# Simplified Template Families mapped by (Intent, Severity, SyntacticPattern)
# Designed to remain strictly within the 10-25 word boundary for authentic PSAs.

DOMAIN_OPENINGS = {
    "Education": ["Ministry Notice:", "Official Advisory:", "Education Notice:", "Public Notice:"],
    "Agriculture": ["Farming Notice:", "Agricultural Alert:", "Official Advisory:"],
    "Health": ["Health Notice:", "Medical Advisory:", "Public Health Alert:"],
    "Security & Safety": ["Security Alert:", "Safety Notice:", "Official Advisory:"],
    "Governance": ["Public Notice:", "Official Advisory:", "Government Notice:"]
}

TEMPLATE_FAMILIES = {
    "Warning": {
        "Emergency": {
            "Imperative": [
                "{opening} Immediately {action_imperative} {location} due to {hazard}!",
                "{opening} {action_imperative} immediately to protect against {hazard}."
            ],
            "Declarative": [
                "{opening} {institution} directs all {audience} to immediately {action_infinitive}."
            ],
            "Passive": [
                "{opening} {action_noun} is urgently required for all {audience} {location}."
            ],
            "Conditional": [
                "{opening} If facing {hazard}, {audience} must immediately {action_infinitive}."
            ]
        },
        "Warning": {
            "Imperative": [
                "{opening} {action_imperative} {location} to avoid {hazard}.",
                "{opening} {action_imperative} to protect yourself from {hazard}."
            ],
            "Declarative": [
                "{opening} {institution} directs {audience} to {action_infinitive}."
            ],
            "Passive": [
                "{opening} {action_noun} is advised {location} to safeguard {audience}."
            ],
            "Conditional": [
                "{opening} If facing {hazard}, {audience} should {action_infinitive}."
            ]
        }
    },
    "Reminder": {
        "Routine": {
            "Imperative": [
                "{opening} Remember to {action_infinitive} before the deadline.",
                "{opening} {action_imperative} before the deadline."
            ],
            "Declarative": [
                "{opening} {institution} reminds {audience} to {action_infinitive}."
            ],
            "Passive": [
                "{opening} {action_noun} is required from {audience} before the deadline."
            ],
            "Conditional": [
                "{opening} If eligible, {audience} should {action_infinitive}."
            ]
        }
    },
    "Advice": {
        "Routine": {
            "Imperative": [
                "{opening} Always {action_imperative} for compliance.",
                "{opening} {action_imperative} to follow procedures."
            ],
            "Declarative": [
                "{opening} {institution} advises {audience} to {action_infinitive}."
            ],
            "Passive": [
                "{opening} {action_noun} is highly recommended for {audience}."
            ],
            "Conditional": [
                "{opening} If you are {audience}, you are encouraged to {action_infinitive}."
            ]
        }
    }
}
