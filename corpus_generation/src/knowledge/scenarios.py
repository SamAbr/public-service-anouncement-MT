from .entities import Institution, Hazard, Audience, Location, Entity, RelationshipConstraint, Action
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass(frozen=True)
class Scenario:
    id: str
    domain: str
    topic: str
    subtopic: str
    allowed_goals: List[str]
    institutions: List[Institution]
    audiences: List[Audience]
    hazards: List[Hazard]
    locations: List[Location]
    actions: List[Action]
    terminology: List[str]
    relationships: List[RelationshipConstraint]
    allowed_seasons: List[str] = field(default_factory=lambda: ["any"])

# --- Define Reusable Entities ---
INSTITUTIONS = {
    "KNEC": Institution("KNEC", "Kenya National Examinations Council (KNEC)", type="Regulatory Body", allowed_domains=["Education"]),
    "HELB": Institution("HELB", "Higher Education Loans Board (HELB)", type="Regulatory Body", allowed_domains=["Education"]),
    "TSC": Institution("TSC", "Teachers Service Commission (TSC)", type="Regulatory Body", allowed_domains=["Education"]),
    "KUCCPS": Institution("KUCCPS", "Kenya Universities and Colleges Central Placement Service (KUCCPS)", type="Regulatory Body", allowed_domains=["Education"]),
    "MoE": Institution("MoE", "Ministry of Education", type="National Government", allowed_domains=["Education"]),
    "MoA": Institution("MoA", "Ministry of Agriculture and Livestock Development", type="National Government", allowed_domains=["Agriculture"]),
    "KEPHIS": Institution("KEPHIS", "Kenya Plant Health Inspectorate Service (KEPHIS)", type="Regulatory Body", allowed_domains=["Agriculture"]),
    "KALRO": Institution("KALRO", "Kenya Agricultural and Livestock Research Organisation (KALRO)", type="Research Body", allowed_domains=["Agriculture"]),
    "NDMA": Institution("NDMA", "National Drought Management Authority (NDMA)", type="Emergency Services", allowed_domains=["Agriculture", "Security & Safety"]),
    "NTSA": Institution("NTSA", "National Transport and Safety Authority (NTSA)", type="Regulatory Body", allowed_domains=["Security & Safety"]),
    "NPS": Institution("NPS", "National Police Service", type="Emergency Services", allowed_domains=["Security & Safety"]),
    "DCI": Institution("DCI", "Directorate of Criminal Investigations (DCI)", type="Emergency Services", allowed_domains=["Security & Safety"]),
    "NC4": Institution("NC4", "National Computer and Cybercrimes Coordination Committee (NC4)", type="Regulatory Body", allowed_domains=["Security & Safety"]),
    "KRA": Institution("KRA", "Kenya Revenue Authority (KRA)", type="Regulatory Body", allowed_domains=["Governance"]),
    "EACC": Institution("EACC", "Ethics and Anti-Corruption Commission (EACC)", type="Regulatory Body", allowed_domains=["Governance"]),
    "ODPC": Institution("ODPC", "Office of the Data Protection Commissioner (ODPC)", type="Regulatory Body", allowed_domains=["Governance"]),
    "MoH": Institution("MoH", "Ministry of Health", type="National Government", allowed_domains=["Health"]),
    "SHA": Institution("SHA", "Social Health Authority (SHA)", type="Regulatory Body", allowed_domains=["Health"]),
    "PPB": Institution("PPB", "Pharmacy and Poisons Board (PPB)", type="Regulatory Body", allowed_domains=["Health"])
}

AUDIENCES = {
    "candidates": Audience("candidates", "candidates preparing for national examinations", allowed_domains=["Education"]),
    "headteachers": Audience("headteachers", "primary and secondary school headteachers", allowed_domains=["Education"]),
    "teachers": Audience("teachers", "newly recruited teachers and educators", allowed_domains=["Education"]),
    "students": Audience("students", "university students and loan applicants", allowed_domains=["Education"]),
    "farmers": Audience("farmers", "smallholder farmers and pastoralists", allowed_domains=["Agriculture"]),
    "drivers": Audience("drivers", "motorists and public service vehicle drivers", allowed_domains=["Security & Safety"]),
    "citizens": Audience("citizens", "members of the general public", allowed_domains=["Security & Safety", "Governance", "Health"]),
    "taxpayers": Audience("taxpayers", "registered taxpayers and business owners", allowed_domains=["Governance"]),
    "patients": Audience("patients", "patients and healthcare consumers", allowed_domains=["Health"])
}

HAZARDS = {
    "cheating": Hazard("cheating", "increasing cases of examination malpractices and fake leakages", related_seasons=["any"]),
    "deadline_exam": Hazard("deadline_exam", "the fast-approaching registration deadline", related_seasons=["any"]),
    "delays_loans": Hazard("delays_loans", "potential delays in disbursement of student bursaries", related_seasons=["any"]),
    "fake_ads": Hazard("fake_ads", "misleading placement advertisements by unregistered private colleges", related_seasons=["any"]),
    "locusts": Hazard("locusts", "the spread of crop diseases and invasive pests like fall armyworms", related_seasons=["dry", "rainy"]),
    "drought": Hazard("drought", "severe drought conditions causing pasture and crop failure", related_seasons=["dry"]),
    "floods": Hazard("floods", "ongoing heavy rains causing severe flash floods on major roads", related_seasons=["rainy"]),
    "phishing": Hazard("phishing", "rising incidents of online phishing scams and identity theft", related_seasons=["any"]),
    "accidents": Hazard("accidents", "increased road accidents due to reckless driving", related_seasons=["any"]),
    "tax_penalties": Hazard("tax_penalties", "penalties for failing to file annual tax returns before the deadline", related_seasons=["any"]),
    "corruption": Hazard("corruption", "fraudulent individuals impersonating government officers to solicit bribes", related_seasons=["any"]),
    "data_breaches": Hazard("data_breaches", "unauthorized data breaches exposing sensitive personal records", related_seasons=["any"]),
    "cholera": Hazard("cholera", "the outbreak of cholera and other waterborne diseases in the area", related_seasons=["rainy"]),
    "fake_drugs": Hazard("fake_drugs", "the distribution of counterfeit medicines and illegal pharmaceutical products", related_seasons=["any"]),
    "insurance_deadline": Hazard("insurance_deadline", "the transition deadline for registration in the social health insurance program", related_seasons=["any"])
}

LOCATIONS = {
    "exam_centers": Location("exam_centers", "at their respective school centers", type="Region"),
    "subcounty": Location("subcounty", "at the nearest sub-county education offices", type="Region"),
    "uni_portal": Location("uni_portal", "online via the student portal", type="Virtual"),
    "web_portal": Location("web_portal", "through the web portal", type="Virtual"),
    "farm_regions": Location("farm_regions", "across the regional agricultural hubs", type="Region"),
    "highways": Location("highways", "on all major national highways", type="Region"),
    "police_hotline": Location("police_hotline", "via the police helpline or local stations", type="Virtual"),
    "tax_portal": Location("tax_portal", "online via the iTax portal", type="Virtual"),
    "huduma": Location("huduma", "at Huduma Centers countrywide", type="Region"),
    "health_centers": Location("health_centers", "at the nearest public health facilities", type="Region"),
    "pharmacies": Location("pharmacies", "from registered retail chemist outlets", type="Region")
}

# --- Define Scenarios by Domain ---
SCENARIOS = {
    "Education": [
        Scenario(
            id="exam_security",
            domain="Education",
            topic="Examinations",
            subtopic="Malpractice",
            allowed_goals=["enforce_integrity", "report_cheating"],
            institutions=[INSTITUTIONS["KNEC"], INSTITUTIONS["MoE"]],
            audiences=[AUDIENCES["candidates"], AUDIENCES["headteachers"]],
            hazards=[HAZARDS["cheating"], HAZARDS["deadline_exam"]],
            locations=[LOCATIONS["exam_centers"], LOCATIONS["subcounty"]],
            actions=[
                Action("verify_reg", "verify registration details", infinitive="[verify] their registration details online", imperative="[verify] your registration details online", noun="Registration [verify]"),
                Action("adhere_guidelines", "adhere to guidelines", infinitive="[adhere] to strict examination guidelines", imperative="[adhere] to strict examination guidelines", noun="[adhere] to examination guidelines"),
                Action("report_leakage", "report exam leakage", infinitive="[report] any instances of exam leakage or cheating", imperative="[report] any instances of exam leakage or cheating", noun="[report] exam leakage")
            ],
            terminology=["examination guidelines", "integrity protocols", "exam registration"],
            relationships=[
                RelationshipConstraint("KNEC", ["candidates"], ["verify_reg", "adhere_guidelines"], ["deadline_exam", "cheating"]),
                RelationshipConstraint("MoE", ["headteachers"], ["report_leakage", "adhere_guidelines"], ["cheating"])
            ]
        ),
        Scenario(
            id="student_funding",
            domain="Education",
            topic="Higher Education",
            subtopic="Funding and Loans",
            allowed_goals=["submit_applications", "manage_delays"],
            institutions=[INSTITUTIONS["HELB"], INSTITUTIONS["MoE"]],
            audiences=[AUDIENCES["students"], AUDIENCES["candidates"]],
            hazards=[HAZARDS["delays_loans"]],
            locations=[LOCATIONS["uni_portal"], LOCATIONS["subcounty"]],
            actions=[
                Action("submit_loan", "submit loan applications", infinitive="[submit] their loan and scholarship applications on time", imperative="[submit] your loan and scholarship applications on time", noun="Timely [submit] of loan applications"),
                Action("update_bank", "update bank details", infinitive="[update] their bank account and disbursement details", imperative="[update] your bank account and disbursement details", noun="[update] bank details")
            ],
            terminology=["tuition loans", "bursary disbursement", "scholarship models"],
            relationships=[
                RelationshipConstraint("HELB", ["students"], ["submit_loan", "update_bank"], ["delays_loans"])
            ]
        ),
        Scenario(
            id="university_placement",
            domain="Education",
            topic="Higher Education",
            subtopic="Placement",
            allowed_goals=["update_profiles", "prevent_fraud"],
            institutions=[INSTITUTIONS["KUCCPS"], INSTITUTIONS["MoE"]],
            audiences=[AUDIENCES["candidates"], AUDIENCES["students"]],
            hazards=[HAZARDS["fake_ads"]],
            locations=[LOCATIONS["web_portal"]],
            actions=[
                Action("update_profiles", "update placement profiles", infinitive="[update] their institution placement profiles through the web portal", imperative="[update] your institution placement profiles through the web portal", noun="[update] placement profiles"),
                Action("verify_colleges", "verify college accreditation", infinitive="[verify] the accreditation status of private colleges", imperative="[verify] the accreditation status of private colleges", noun="[verify] of college accreditation status")
            ],
            terminology=["accreditation portal", "placement indexes", "admission letters"],
            relationships=[
                RelationshipConstraint("KUCCPS", ["candidates"], ["update_profiles"], ["fake_ads"])
            ]
        )
    ],
    "Agriculture": [
        Scenario(
            id="pest_control",
            domain="Agriculture",
            topic="Crop Production",
            subtopic="Pest Management",
            allowed_goals=["pest_reporting", "crop_protection"],
            institutions=[INSTITUTIONS["MoA"], INSTITUTIONS["KEPHIS"]],
            audiences=[AUDIENCES["farmers"]],
            hazards=[HAZARDS["locusts"]],
            locations=[LOCATIONS["farm_regions"]],
            actions=[
                Action("report_pests", "report crop pests", infinitive="[report] cases of locusts or armyworms immediately", imperative="[report] cases of locusts or armyworms immediately", noun="[report] crop pests"),
                Action("spray_crops", "spray crop fields", infinitive="[apply] approved chemical sprays to affected fields", imperative="[apply] approved chemical sprays to affected fields", noun="[apply] chemical sprays to crop fields")
            ],
            terminology=["certified pesticides", "extension support", "crop inspection"],
            relationships=[
                RelationshipConstraint("KEPHIS", ["farmers"], ["report_pests"], ["locusts"]),
                RelationshipConstraint("MoA", ["farmers"], ["spray_crops"], ["locusts"])
            ]
        ),
        Scenario(
            id="drought_resilience",
            domain="Agriculture",
            topic="Livestock",
            subtopic="Drought Management",
            allowed_goals=["harvest_water", "feed_preservation"],
            institutions=[INSTITUTIONS["MoA"], INSTITUTIONS["NDMA"]],
            audiences=[AUDIENCES["farmers"]],
            hazards=[HAZARDS["drought"]],
            locations=[LOCATIONS["farm_regions"]],
            actions=[
                Action("harvest_water", "harvest rainwater", infinitive="[adopt] modern water harvesting techniques", imperative="[adopt] modern water harvesting techniques", noun="[adopt] of water harvesting techniques"),
                Action("store_hay", "store hay and feed", infinitive="[stockpile] hay and supplementary feeds", imperative="[stockpile] hay and supplementary feeds", noun="[stockpile] hay and supplementary feeds")
            ],
            terminology=["pasture preservation", "livestock feeds", "water pans"],
            relationships=[
                RelationshipConstraint("NDMA", ["farmers"], ["harvest_water", "store_hay"], ["drought"])
            ],
            allowed_seasons=["dry"]
        )
    ],
    "Security & Safety": [
        Scenario(
            id="cyber_security",
            domain="Security & Safety",
            topic="Digital Security",
            subtopic="Cyber Fraud",
            allowed_goals=["protect_credentials", "report_scams"],
            institutions=[INSTITUTIONS["NC4"], INSTITUTIONS["DCI"]],
            audiences=[AUDIENCES["citizens"]],
            hazards=[HAZARDS["phishing"]],
            locations=[LOCATIONS["web_portal"]],
            actions=[
                Action("enable_2fa", "enable 2FA authentication", infinitive="[enable_2fa] two-factor authentication", imperative="[enable_2fa] two-factor authentication on all accounts", noun="[enable_2fa] two-factor authentication"),
                Action("report_phishing", "report phishing scams", infinitive="[report] suspicious links or phishing messages", imperative="[report] suspicious links or phishing messages", noun="[report] phishing scams")
            ],
            terminology=["2FA authentication", "credentials security", "phishing alerts"],
            relationships=[
                RelationshipConstraint("NC4", ["citizens"], ["enable_2fa", "report_phishing"], ["phishing"])
            ]
        ),
        Scenario(
            id="flood_safety",
            domain="Security & Safety",
            topic="Disaster Safety",
            subtopic="Flooding",
            allowed_goals=["evacuate", "road_safety"],
            institutions=[INSTITUTIONS["NPS"], INSTITUTIONS["NDMA"]],
            audiences=[AUDIENCES["drivers"], AUDIENCES["citizens"]],
            hazards=[HAZARDS["floods"]],
            locations=[LOCATIONS["highways"]],
            actions=[
                Action("avoid_crossing", "avoid flooded bridges", infinitive="[avoid] crossing flooded bridges or swollen rivers", imperative="[avoid] crossing flooded bridges or swollen rivers", noun="[avoid] flooded crossings"),
                Action("move_higher", "relocate to higher ground", infinitive="[relocate] to higher ground away from riverbeds", imperative="[relocate] to higher ground away from riverbeds", noun="[relocate] to higher ground")
            ],
            terminology=["flash flood warnings", "emergency assistance", "river levels"],
            relationships=[
                RelationshipConstraint("NPS", ["drivers"], ["avoid_crossing"], ["floods"]),
                RelationshipConstraint("NDMA", ["citizens"], ["move_higher"], ["floods"])
            ],
            allowed_seasons=["rainy"]
        ),
        Scenario(
            id="road_safety",
            domain="Security & Safety",
            topic="Transport Safety",
            subtopic="Traffic Compliance",
            allowed_goals=["comply_speed", "vehicle_check"],
            institutions=[INSTITUTIONS["NTSA"], INSTITUTIONS["NPS"]],
            audiences=[AUDIENCES["drivers"]],
            hazards=[HAZARDS["accidents"]],
            locations=[LOCATIONS["highways"]],
            actions=[
                Action("check_mechanics", "check vehicle mechanics", infinitive="[verify] the mechanical condition of their vehicles", imperative="[verify] the mechanical condition of your vehicle", noun="[verify] vehicle mechanical condition"),
                Action("obey_limits", "obey speed limits", infinitive="[adhere] with speed limits on all expressways", imperative="[adhere] with speed limits on all expressways", noun="[adhere] with speed limits")
            ],
            terminology=["speed governors", "roadworthiness audits", "mechanical checks"],
            relationships=[
                RelationshipConstraint("NTSA", ["drivers"], ["check_mechanics", "obey_limits"], ["accidents"])
            ]
        )
    ],
    "Governance": [
        Scenario(
            id="tax_compliance",
            domain="Governance",
            topic="Public Finance",
            subtopic="Taxation",
            allowed_goals=["file_returns", "avoid_penalties"],
            institutions=[INSTITUTIONS["KRA"]],
            audiences=[AUDIENCES["taxpayers"]],
            hazards=[HAZARDS["tax_penalties"]],
            locations=[LOCATIONS["tax_portal"]],
            actions=[
                Action("file_early", "file tax returns", infinitive="[submit] their annual tax returns early", imperative="[submit] your annual tax returns early", noun="Early [submit] of annual tax returns")
            ],
            terminology=["iTax credentials", "penalty waivers", "tax compliance certificates"],
            relationships=[
                RelationshipConstraint("KRA", ["taxpayers"], ["file_early"], ["tax_penalties"])
            ]
        ),
        Scenario(
            id="anti_corruption",
            domain="Governance",
            topic="Integrity",
            subtopic="Bribery",
            allowed_goals=["report_bribes", "verify_officials"],
            institutions=[INSTITUTIONS["EACC"], INSTITUTIONS["NPS"]],
            audiences=[AUDIENCES["citizens"]],
            hazards=[HAZARDS["corruption"]],
            locations=[LOCATIONS["huduma"], LOCATIONS["police_hotline"]],
            actions=[
                Action("report_bribery", "report bribes", infinitive="[report] public officials demanding bribes", imperative="[report] public officials demanding bribes", noun="[report] corruption and bribery"),
                Action("demand_id", "verify officer identity", infinitive="[verify] the identity badges of officers", imperative="[verify] the identity badges of officers", noun="[verify] officer identity")
            ],
            terminology=["corruption hotlines", "whistleblower protection", "bribe reporting"],
            relationships=[
                RelationshipConstraint("EACC", ["citizens"], ["report_bribery"], ["corruption"])
            ]
        ),
        Scenario(
            id="data_protection",
            domain="Governance",
            topic="Human Rights",
            subtopic="Privacy",
            allowed_goals=["secure_records", "verify_access"],
            institutions=[INSTITUTIONS["ODPC"]],
            audiences=[AUDIENCES["citizens"]],
            hazards=[HAZARDS["data_breaches"]],
            locations=[LOCATIONS["web_portal"]],
            actions=[
                Action("secure_pins", "secure PINs and passwords", infinitive="[avoid] sharing personal identification numbers or passwords", imperative="[avoid] sharing personal identification numbers or passwords", noun="[avoid] sharing personal PINs and passwords"),
                Action("report_breach", "report data breaches", infinitive="[report] unauthorized data sharing incidents", imperative="[report] unauthorized data sharing incidents", noun="[report] data privacy breaches")
            ],
            terminology=["data breaches", "personal passwords", "privacy audits"],
            relationships=[
                RelationshipConstraint("ODPC", ["citizens"], ["secure_pins", "report_breach"], ["data_breaches"])
            ]
        )
    ],
    "Health": [
        Scenario(
            id="waterborne_outbreak",
            domain="Health",
            topic="Public Health",
            subtopic="Sanitation",
            allowed_goals=["boil_water", "hygiene_check"],
            institutions=[INSTITUTIONS["MoH"]],
            audiences=[AUDIENCES["citizens"]],
            hazards=[HAZARDS["cholera"]],
            locations=[LOCATIONS["health_centers"]],
            actions=[
                Action("boil_water", "boil drinking water", infinitive="[boil] all drinking water and maintain strict hygiene", imperative="[boil] all drinking water and maintain strict hygiene", noun="[boil] drinking water and sanitation hygiene"),
                Action("visit_clinic", "visit health clinic", infinitive="[relocate] to the nearest public health clinic immediately if symptomatic", imperative="[relocate] to the nearest public health clinic immediately if you feel symptomatic", noun="[relocate] to public health clinics")
            ],
            terminology=["rehydration salts", "water treatment", "chlorine tablets"],
            relationships=[
                RelationshipConstraint("MoH", ["citizens"], ["boil_water", "visit_clinic"], ["cholera"])
            ],
            allowed_seasons=["rainy"]
        ),
        Scenario(
            id="health_insurance",
            domain="Health",
            topic="Universal Health Coverage",
            subtopic="Insurance",
            allowed_goals=["register_shif", "verify_status"],
            institutions=[INSTITUTIONS["SHA"], INSTITUTIONS["MoH"]],
            audiences=[AUDIENCES["citizens"]],
            hazards=[HAZARDS["insurance_deadline"]],
            locations=[LOCATIONS["huduma"], LOCATIONS["web_portal"]],
            actions=[
                Action("register_shif", "register for SHA", infinitive="[register] for the social health insurance fund", imperative="[register] for the social health insurance fund", noun="[register] for social health insurance"),
                Action("verify_status", "verify registration status", infinitive="[verify] registration status on the citizen portal", imperative="[verify] your registration status on the citizen portal", noun="[verify] of registration status")
            ],
            terminology=["SHIF registrations", "household coverages", "member portals"],
            relationships=[
                RelationshipConstraint("SHA", ["citizens"], ["register_shif", "verify_status"], ["insurance_deadline"])
            ]
        ),
        Scenario(
            id="safe_medication",
            domain="Health",
            topic="Medical Safety",
            subtopic="Counterfeit Drugs",
            allowed_goals=["check_seals", "report_unlicensed"],
            institutions=[INSTITUTIONS["PPB"], INSTITUTIONS["MoH"]],
            audiences=[AUDIENCES["patients"], AUDIENCES["citizens"]],
            hazards=[HAZARDS["fake_drugs"]],
            locations=[LOCATIONS["pharmacies"]],
            actions=[
                Action("check_seals", "verify safety seals", infinitive="[verify] safety seals and registration numbers on packaging", imperative="[verify] safety seals and registration numbers on packaging", noun="[verify] drug safety seals"),
                Action("report_unlicensed", "report unlicensed pharmacies", infinitive="[report] unlicensed drug retailers operating in local markets", imperative="[report] unlicensed drug retailers operating in local markets", noun="[report] unlicensed retail chemists")
            ],
            terminology=["pharmacy registration locks", "drug safety approvals", "counterfeit medicines"],
            relationships=[
                RelationshipConstraint("PPB", ["patients", "citizens"], ["check_seals", "report_unlicensed"], ["fake_drugs"])
            ]
        )
    ]
}

def get_scenarios_for_domain(domain: str) -> List[Scenario]:
    return SCENARIOS.get(domain, [])
