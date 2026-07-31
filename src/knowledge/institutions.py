INSTITUTIONS = {
    "Education": [
        ("Ministry of Education", {"exams", "loans", "placement", "teachers", "general"}),
        ("Kenya National Examinations Council (KNEC)", {"exams"}),
        ("Kenya Universities and Colleges Central Placement Service (KUCCPS)", {"placement"}),
        ("Higher Education Loans Board (HELB)", {"loans"}),
        ("Teachers Service Commission (TSC)", {"teachers"}),
        ("Kenya Institute of Curriculum Development (KICD)", {"general"})
    ],
    "Agriculture": [
        ("Ministry of Agriculture and Livestock Development", {"crops", "livestock", "pests", "general"}),
        ("Kenya Plant Health Inspectorate Service (KEPHIS)", {"crops", "pests"}),
        ("Kenya Agricultural and Livestock Research Organisation (KALRO)", {"crops", "livestock"}),
        ("Agriculture and Food Authority (AFA)", {"crops", "general"}),
        ("National Drought Management Authority (NDMA)", {"weather"}),
        ("Kenya Meteorological Department", {"weather"})
    ],
    "Security & Safety": [
        ("National Police Service", {"safety", "general"}),
        ("Directorate of Criminal Investigations (DCI)", {"safety"}),
        ("National Transport and Safety Authority (NTSA)", {"road"}),
        ("National Disaster Management Unit (NDMU)", {"floods", "weather"}),
        ("National Computer and Cybercrimes Coordination Committee (NC4)", {"cyber"}),
        ("Kenya Red Cross Society", {"floods", "weather", "general"})
    ],
    "Governance": [
        ("eCitizen Portal Administration", {"identity", "general"}),
        ("Huduma Kenya Secretariat", {"identity"}),
        ("Kenya Revenue Authority (KRA)", {"tax"}),
        ("Central Bank of Kenya (CBK)", {"general"}),
        ("Office of the Data Protection Commissioner (ODPC)", {"data"}),
        ("Ethics and Anti-Corruption Commission (EACC)", {"corruption"})
    ],
    "Health": [
        ("Ministry of Health", {"cholera", "insurance", "drugs", "general"}),
        ("Social Health Authority (SHA)", {"insurance"}),
        ("Kenya Medical Supplies Authority (KEMSA)", {"general"}),
        ("Pharmacy and Poisons Board (PPB)", {"drugs"}),
        ("National Syndemic Diseases Control Council (NSDCC)", {"general"}),
        ("Kenya Medical Research Institute (KEMRI)", {"general"})
    ]
}

def get_institutions_for_domain(domain):
    return INSTITUTIONS.get(domain, [])
