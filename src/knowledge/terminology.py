TERMINOLOGY = {
    "Education": [
        "competency-based curriculum (CBC)",
        "higher education funding model",
        "national examination council guidelines",
        "sub-county education offices",
        "university placement portal"
    ],
    "Agriculture": [
        "certified seed varieties",
        "agricultural extension officers",
        "post-harvest management",
        "livestock vaccination campaigns",
        "drip irrigation kits"
    ],
    "Security & Safety": [
        "road safety compliance",
        "cybersecurity hygiene practices",
        "emergency response hotlines",
        "disaster management protocols",
        "public safety awareness"
    ],
    "Governance": [
        "citizen self-service portal",
        "annual tax returns filing",
        "digital service delivery",
        "anti-corruption commission",
        "data protection regulations"
    ],
    "Health": [
        "community health promoters",
        "scheduled immunization programs",
        "water sanitation and hygiene (WASH)",
        "preventive healthcare practices",
        "universal health coverage"
    ]
}

def get_terminology_for_domain(domain):
    return TERMINOLOGY.get(domain, [])
