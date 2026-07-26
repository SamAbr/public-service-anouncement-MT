INSTITUTIONS = {
    "Education": [
        "Ministry of Education",
        "Kenya National Examinations Council (KNEC)",
        "Kenya Universities and Colleges Central Placement Service (KUCCPS)",
        "Higher Education Loans Board (HELB)",
        "Teachers Service Commission (TSC)",
        "Kenya Institute of Curriculum Development (KICD)"
    ],
    "Agriculture": [
        "Ministry of Agriculture and Livestock Development",
        "Kenya Plant Health Inspectorate Service (KEPHIS)",
        "Kenya Agricultural and Livestock Research Organisation (KALRO)",
        "Agriculture and Food Authority (AFA)",
        "National Drought Management Authority (NDMA)",
        "Kenya Meteorological Department"
    ],
    "Security & Safety": [
        "National Police Service",
        "Directorate of Criminal Investigations (DCI)",
        "National Transport and Safety Authority (NTSA)",
        "National Disaster Management Unit (NDMU)",
        "National Computer and Cybercrimes Coordination Committee (NC4)",
        "Kenya Red Cross Society"
    ],
    "Governance": [
        "eCitizen Portal Administration",
        "Huduma Kenya Secretariat",
        "Kenya Revenue Authority (KRA)",
        "Central Bank of Kenya (CBK)",
        "Office of the Data Protection Commissioner (ODPC)",
        "Ethics and Anti-Corruption Commission (EACC)"
    ],
    "Health": [
        "Ministry of Health",
        "Social Health Authority (SHA)",
        "Kenya Medical Supplies Authority (KEMSA)",
        "Pharmacy and Poisons Board (PPB)",
        "National Syndemic Diseases Control Council (NSDCC)",
        "Kenya Medical Research Institute (KEMRI)"
    ]
}

def get_institutions_for_domain(domain):
    return INSTITUTIONS.get(domain, [])
