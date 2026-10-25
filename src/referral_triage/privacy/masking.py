from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.predefined_recognizers import PhoneRecognizer

from referral_triage.privacy.nhi import is_valid_nhi

from presidio_analyzer.nlp_engine import NlpEngineProvider

ENTITIES = ["NZ_NHI", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "DATE_TIME", "LOCATION"]


class NHIRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="NZ_NHI",
            patterns=[Pattern("nhi_old", r"\b[A-HJ-NP-Z]{3}\d{4}\b", 0.4),
                      Pattern("nhi_new", r"\b[A-HJ-NP-Z]{3}\d{2}[A-HJ-NP-Z]{2}\b", 0.4)],
            context=["nhi", "national health index"],
        )

    def validate_result(self, pattern_text: str):
        return is_valid_nhi(pattern_text)      # invalid check digit -> not flagged


def build_analyzer() -> AnalyzerEngine:
    model = os.getenv("SPACY_MODEL", "en_core_web_lg")
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": model}],
    }).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    analyzer.registry.add_recognizer(NHIRecognizer())
    analyzer.registry.add_recognizer(PhoneRecognizer(supported_regions=["NZ"]))
    return analyzer


def mask(text: str, analyzer: AnalyzerEngine, known_names: list[str] | None = None):
    """Replace PII with numbered tokens. Returns (masked_text, token_map)."""
    extra = []
    if known_names:   # names from the FHIR Patient record - catches names spaCy misses
        extra.append(PatternRecognizer(supported_entity="PERSON", deny_list=known_names,
                                       name="case_names"))
    results = analyzer.analyze(text=text, entities=ENTITIES, language="en",
                               score_threshold=0.35, ad_hoc_recognizers=extra)
    kept = []         # resolve overlaps: highest score, then longest span
    for r in sorted(results, key=lambda r: (-r.score, -(r.end - r.start))):
        if all(r.end <= k.start or r.start >= k.end for k in kept):
            kept.append(r)
    token_map, counts, out = {}, {}, text
    for r in sorted(kept, key=lambda r: r.start, reverse=True):
        counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
        token = f"<{r.entity_type}_{counts[r.entity_type]}>"
        token_map[token] = text[r.start:r.end]
        out = out[:r.start] + token + out[r.end:]
    return out, token_map


def unmask(text: str, token_map: dict) -> str:
    for token, original in token_map.items():
        text = text.replace(token, original)
    return text