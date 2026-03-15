SECTION_MAP = {
    "cause": "causes",
    "causes": "causes",
    "symptom": "symptoms",
    "symptoms": "symptoms",
    "diagnosis": "diagnosis",
    "diagnose": "diagnosis",
    "treat": "treatment",
    "treatment": "treatment",
    "therapy": "treatment",
    "prevent": "prevention",
    "prevention": "prevention",
    "risk": "risks",
    "prognosis": "prognosis",
}


def detect_section(query: str):
    q = query.lower()

    for word, section in SECTION_MAP.items():
        if word in q:
            return section

    return None