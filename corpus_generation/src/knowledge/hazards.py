HAZARDS = {
    "Education": [
        ("the fast-approaching deadline for national exam registrations", {"exams"}),
        ("increasing cases of examination malpractices and fake leakages", {"exams"}),
        ("unauthorized fee increases and illegal levies in public schools", {"general"}),
        ("misleading placement advertisements by unregistered private colleges", {"placement"}),
        ("potential delays in disbursement of student bursaries and tuition loans", {"loans"})
    ],
    "Agriculture": [
        ("the spread of crop diseases and invasive pests across the region", {"crops", "pests"}),
        ("severe drought conditions causing pasture and crop failure", {"weather"}),
        ("the circulation of counterfeit seeds and substandard fertilizers in the market", {"crops"}),
        ("high risk of post-harvest losses and aflatoxin contamination due to dampness", {"crops"}),
        ("unpredictable weather patterns and delayed onset of seasonal rains", {"weather", "crops"})
    ],
    "Security & Safety": [
        ("rising incidents of online phishing scams and identity theft", {"cyber"}),
        ("the ongoing heavy rains causing severe flash floods on major roads", {"floods", "weather"}),
        ("increased road accidents due to reckless driving and poor vehicle maintenance", {"road"}),
        ("heightened security threats in crowded public venues and social spaces", {"safety"}),
        ("wildfires and fire hazards in residential areas during dry spells", {"weather", "safety"})
    ],
    "Governance": [
        ("penalties for failing to file annual tax returns before the deadline", {"tax"}),
        ("fraudulent individuals impersonating government officers to solicit bribes", {"corruption"}),
        ("unauthorized data breaches exposing sensitive personal records", {"data"}),
        ("a rise in fake certificates and identity theft online", {"identity"}),
        ("the official deadline for registering businesses and property listings", {"general"})
    ],
    "Health": [
        ("the outbreak of cholera and other waterborne diseases in the area", {"cholera"}),
        ("the transition deadline for registration in the social health insurance program", {"insurance"}),
        ("the distribution of counterfeit medicines and illegal pharmaceutical products", {"drugs"}),
        ("rising cases of seasonal respiratory infections and flu", {"general"}),
        ("delayed medical attention for treatable chronic conditions", {"general"})
    ]
}

def get_hazards_for_domain(domain):
    return HAZARDS.get(domain, [])
