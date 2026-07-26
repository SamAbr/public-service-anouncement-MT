LOCATIONS = {
    "Education": [
        "across all primary and secondary schools in the country",
        "in all registered public and private universities",
        "at the nearest sub-county education offices",
        "within various regional assessment and training centers",
        "online via the official web portal"
    ],
    "Agriculture": [
        "in all agricultural counties and farming communities",
        "at local national cereals and produce board depots",
        "across arid and semi-arid lands of Northern Kenya",
        "at KEPHIS and KALRO regional stations",
        "within major grain-producing zones in the Rift Valley"
    ],
    "Security & Safety": [
        "on all major highways and public transit routes",
        "in flood-prone regions and low-lying river valleys",
        "at all public gatherings and commercial centers",
        "in residential estates and urban neighborhoods",
        "near police stations and administrative posts"
    ],
    "Governance": [
        "at all Huduma Centres across the forty-seven counties",
        "online via the citizen self-service portal",
        "at KRA offices and custom border points",
        "within all government registries and public offices",
        "in all sub-county administrative headquarters"
    ],
    "Health": [
        "in all public dispensaries, clinics, and hospitals",
        "at community health outreach units nationwide",
        "within residential areas experiencing sanitation issues",
        "at designated vaccination and immunization desks",
        "in markets, restaurants, and food preparation premises"
    ]
}

def get_locations_for_domain(domain):
    return LOCATIONS.get(domain, [])
