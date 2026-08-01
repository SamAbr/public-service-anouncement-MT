# Simplified Template Families mapped by (Intent, Severity, SyntacticPattern)
# Designed to remain strictly within the 10-25 word boundary for authentic PSAs.
# Adjusted to be 100% active, punchy, and free of clunky conditional 'If' or passive voice subjects.

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
                "{opening} {institution} urgently requires {audience} to {action_infinitive} {location}."
            ],
            "Conditional": [
                "{opening} {audience} must immediately {action_infinitive} when facing {hazard}."
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
                "{opening} {institution} advises {audience} to {action_infinitive} to safeguard against {hazard}."
            ],
            "Conditional": [
                "{opening} {audience} should {action_infinitive} when facing {hazard}."
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
                "{opening} {institution} requires {audience} to {action_infinitive} before the deadline."
            ],
            "Conditional": [
                "{opening} Eligible {audience} are requested to {action_infinitive}."
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
                "{opening} {institution} highly recommends {audience} to {action_infinitive}."
            ],
            "Conditional": [
                "{opening} {audience} are encouraged to {action_infinitive}."
            ]
        }
    }
}
