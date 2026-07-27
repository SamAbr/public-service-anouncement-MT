HAZARDS = {
    "Education": [
        "the fast-approaching deadline for national exam registrations",
        "increasing cases of examination malpractices and fake leakages",
        "unauthorized fee increases and illegal levies in public schools",
        "misleading placement advertisements by unregistered private colleges",
        "potential delays in disbursement of student bursaries and tuition loans"
    ],
    "Agriculture": [
        "the spread of crop diseases and invasive pests across the region",
        "severe drought conditions causing pasture and crop failure",
        "the circulation of counterfeit seeds and substandard fertilizers in the market",
        "high risk of post-harvest losses and aflatoxin contamination due to dampness",
        "unpredictable weather patterns and delayed onset of seasonal rains"
    ],
    "Security & Safety": [
        "rising incidents of online phishing scams and identity theft",
        "the ongoing heavy rains causing severe flash floods on major roads",
        "increased road accidents due to reckless driving and poor vehicle maintenance",
        "heightened security threats in crowded public venues and social spaces",
        "wildfires and fire hazards in residential areas during dry spells"
    ],
    "Governance": [
        "penalties for failing to file annual tax returns before the deadline",
        "fraudulent individuals impersonating government officers to solicit bribes",
        "breaches of personal data privacy by unregulated service providers",
        "unauthorized transactions on digital government payment platforms",
        "delays in renewal of business permits and statutory operating licenses"
    ],
    "Health": [
        "the recent outbreak of cholera in neighboring administrative areas",
        "the resurgence of vaccine-preventable diseases among young children",
        "increased malaria transmission during the rainy season",
        "the sale of counterfeit medicines and unregistered health products",
        "poor food handling practices leading to severe food poisoning incidents"
    ]
}

def get_hazards_for_domain(domain):
    return HAZARDS.get(domain, [])
