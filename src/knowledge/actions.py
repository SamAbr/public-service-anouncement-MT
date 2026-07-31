ACTIONS = {
    "Education": [
        ("verify their registration details online before the deadline", {"exams"}),
        ("submit their loan and scholarship applications on time", {"loans"}),
        ("adhere to the strict examination guidelines to prevent malpractice", {"exams"}),
        ("report any instances of exam leakage or cheating via the official portal", {"exams"}),
        ("enroll for the new competency-based curriculum teacher training workshops", {"teachers"}),
        ("update their institution placement profiles through the web portal", {"placement"})
    ],
    "Agriculture": [
        ("purchase certified crop seeds and fertilizers only from registered dealers", {"crops"}),
        ("vaccinate all their livestock against common seasonal diseases", {"livestock"}),
        ("adopt modern water harvesting and soil conservation techniques", {"weather", "crops"}),
        ("report any cases of invasive pests like locusts or fall armyworms immediately", {"pests"}),
        ("store harvested grain in dry, cool conditions to prevent aflatoxin contamination", {"crops"}),
        ("consult agricultural extension officers regarding current weather forecasts", {"weather", "crops", "livestock"})
    ],
    "Security & Safety": [
        ("exercise extreme caution and avoid crossing flooded bridges or rivers", {"floods", "weather"}),
        ("enable two-factor authentication and secure their online banking credentials", {"cyber"}),
        ("report suspicious activities or abandoned packages to the police hotline", {"safety"}),
        ("check the mechanical condition of their vehicles before embarking on long journeys", {"road"}),
        ("comply with speed limits and road safety rules on all highways", {"road"}),
        ("refrain from sharing unverified news or inflammatory reports online", {"safety", "cyber"})
    ],
    "Governance": [
        ("file their annual tax returns early to avoid the last-minute rush", {"tax"}),
        ("verify their registration status on the official citizen portal", {"identity"}),
        ("report any public officials demanding bribes or illegal processing fees", {"corruption"}),
        ("avoid sharing personal identification numbers or passwords with third parties", {"data"}),
        ("ensure their biometric details are updated at the nearest service office", {"identity"}),
        ("verify the security features of all newly issued currency notes", {"general"})
    ],
    "Health": [
        ("boil all drinking water and maintain strict hygiene standards at home", {"cholera"}),
        ("register for the social health insurance fund to secure medical coverage", {"insurance"}),
        ("verify the safety seals and registration numbers on all purchased medicines", {"drugs"}),
        ("report any unlicensed chemists operating in their local neighborhoods", {"drugs"}),
        ("visit the nearest public health center for free vaccination and checkups", {"general", "cholera"}),
        ("adhere to malaria prevention practices by sleeping under treated mosquito nets", {"general"})
    ]
}

def get_actions_for_domain(domain):
    return ACTIONS.get(domain, [])
