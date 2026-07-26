ACTIONS = {
    "Education": [
        "verify their registration details online before the deadline",
        "submit their loan and scholarship applications on time",
        "adhere to the strict examination guidelines to prevent malpractice",
        "report any instances of exam leakage or cheating via the official portal",
        "enroll for the new competency-based curriculum teacher training workshops",
        "update their institution placement profiles through the web portal"
    ],
    "Agriculture": [
        "purchase certified crop seeds and fertilizers only from registered dealers",
        "vaccinate all their livestock against common seasonal diseases",
        "adopt modern water harvesting and soil conservation techniques",
        "report any cases of invasive pests like locusts or fall armyworms immediately",
        "store harvested grain in dry, cool conditions to prevent aflatoxin contamination",
        "consult agricultural extension officers regarding current weather forecasts"
    ],
    "Security & Safety": [
        "exercise extreme caution and avoid crossing flooded bridges or rivers",
        "enable two-factor authentication and secure their online banking credentials",
        "report suspicious activities or abandoned packages to the police hotline",
        "check the mechanical condition of their vehicles before embarking on long journeys",
        "comply with speed limits and road safety rules on all highways",
        "refrain from sharing unverified news or inflammatory reports online"
    ],
    "Governance": [
        "file their annual income tax returns before the statutory deadline",
        "update their user profiles and contact details on the eCitizen platform",
        "report any corruption attempts or demand for bribes in public offices",
        "register their businesses online to comply with legal regulations",
        "ensure their personal data handles comply with the data protection act",
        "pay their land rates and county permits through designated digital channels"
    ],
    "Health": [
        "boil or treat all drinking water to prevent cholera and other diarrheal diseases",
        "ensure children under five years receive all scheduled immunization vaccines",
        "register for the new national health insurance cover to access subsidized treatment",
        "clear stagnant water around residential plots to eliminate mosquito breeding grounds",
        "maintain high standards of hygiene and wash hands regularly with soap",
        "seek immediate medical attention at the nearest facility if experiencing high fever"
    ]
}

def get_actions_for_domain(domain):
    return ACTIONS.get(domain, [])
