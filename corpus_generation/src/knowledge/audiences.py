AUDIENCES = {
    "Education": [
        "all school-going children, parents, and guardians",
        "primary and secondary school headteachers",
        "university students and helper applicants",
        "candidates preparing for national examinations",
        "newly recruited teachers and educators",
        "members of the general public pursuing higher education"
    ],
    "Agriculture": [
        "smallholder farmers and livestock keepers",
        "maize and wheat farmers in agricultural zones",
        "dairy producers and veterinary officers",
        "agricultural suppliers and distributors",
        "pastoralists in arid and semi-arid regions",
        "all citizens engaged in urban farming initiatives"
    ],
    "Security & Safety": [
        "all motorists, passengers, and pedestrians",
        "online shoppers and social media users",
        "residents living in flood-prone and landslide areas",
        "business owners and financial service agents",
        "public service vehicle (PSV) owners and drivers",
        "all members of the public nationwide"
    ],
    "Governance": [
        "all taxpayers and business proprietors",
        "registered voters and community leaders",
        "citizens seeking government services online",
        "public servants and government ministry officials",
        "data controllers and processor representatives",
        "every citizen residing in Kenya and the diaspora"
    ],
    "Health": [
        "expectant mothers and parents of young children",
        "community health promoters and clinic officers",
        "residents in counties experiencing disease outbreaks",
        "patients relying on public healthcare services",
        "all citizens and residents of Kenya",
        "food handlers and hospitality sector operators"
    ]
}

def get_audiences_for_domain(domain):
    return AUDIENCES.get(domain, [])
