# Template Families mapped by (Intent, Severity, SyntacticPattern)
# Specifically designed for rich syntactic variations in Machine Translation training.

OPENINGS = [
    "Public Announcement:",
    "Ministry Directive:",
    "Press Statement:",
    "Academic Notice:",
    "Official Notice:",
    "Farmers Advisory:",
    "Security Alert:"
]

TEMPLATE_FAMILIES = {
    "Warning": {
        "Emergency": {
            "Imperative": [
                "{opening} Immediately {action_imperative} {location} to protect against {hazard}!",
                "{opening} {action_imperative} {location} without delay to curb the threat of {hazard}."
            ],
            "Declarative": [
                "{opening} The {institution} hereby requires {audience} to immediately {action_infinitive} in response to {hazard} {location}.",
                "{opening} The {institution} advises that all {audience} must urgently {action_infinitive} due to {hazard} {location}."
            ],
            "Passive": [
                "{opening} {action_noun} is strictly required {location} for all {audience} to mitigate {hazard}.",
                "{opening} Immediate {action_noun} is directed {location} to protect all {audience} against {hazard}."
            ],
            "Conditional": [
                "{opening} If you are in {location}, immediately {action_imperative} to avoid the threat of {hazard}.",
                "{opening} If {hazard} is reported {location}, {audience} must immediately {action_infinitive}."
            ]
        },
        "Warning": {
            "Imperative": [
                "{opening} {action_imperative} {location} to protect yourself from {hazard}.",
                "{opening} {action_imperative} {location} in light of {hazard}."
            ],
            "Declarative": [
                "{opening} The {institution} warns {audience} to {action_infinitive} to reduce the risks of {hazard} {location}.",
                "{opening} The {institution} directs all {audience} to {action_infinitive} due to rising cases of {hazard} {location}."
            ],
            "Passive": [
                "{opening} {action_noun} is advised {location} to safeguard {audience} against {hazard}.",
                "{opening} Increased {action_noun} is recommended to protect {audience} from {hazard} {location}."
            ],
            "Conditional": [
                "{opening} If {hazard} affects {location}, {audience} are advised to {action_infinitive}.",
                "{opening} When facing {hazard} {location}, {audience} should promptly {action_infinitive}."
            ]
        }
    },
    "Reminder": {
        "Routine": {
            "Imperative": [
                "{opening} {action_imperative} {location} before the upcoming deadline.",
                "{opening} Remember to {action_infinitive} {location} on time."
            ],
            "Declarative": [
                "{opening} The {institution} reminds all {audience} to {action_infinitive} {location}.",
                "{opening} The {institution} wishes to remind {audience} to {action_infinitive} {location}."
            ],
            "Passive": [
                "{opening} {action_noun} is requested from all {audience} {location} before the deadline.",
                "{opening} Timely {action_noun} is expected from {audience} {location}."
            ],
            "Conditional": [
                "{opening} If you have not done so, please {action_imperative} {location} as required.",
                "{opening} If eligible, {audience} should {action_infinitive} {location}."
            ]
        }
    },
    "Advice": {
        "Routine": {
            "Imperative": [
                "{opening} {action_imperative} {location} for optimal safety and compliance.",
                "{opening} Always {action_imperative} {location} to follow standard procedures."
            ],
            "Declarative": [
                "{opening} The {institution} advises {audience} to {action_infinitive} {location}.",
                "{opening} The {institution} recommends that {audience} {action_infinitive} {location}."
            ],
            "Passive": [
                "{opening} {action_noun} is highly recommended for {audience} {location}.",
                "{opening} {action_noun} is encouraged to ensure better compliance {location}."
            ],
            "Conditional": [
                "{opening} If you want to ensure compliance, {action_imperative} {location}.",
                "{opening} If you are {audience}, you are encouraged to {action_infinitive} {location}."
            ]
        }
    }
}
