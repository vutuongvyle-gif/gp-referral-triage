"""MCP tool server for GP referral triage. Run: make mcp"""
import json

import httpx
import psycopg
from mcp.server.fastmcp import FastMCP

from referral_triage.config import settings
from referral_triage.rag.retriever import format_hits, retrieve

mcp = FastMCP("referral-tools", host=settings.mcp_host, port=settings.mcp_port,
              log_level="WARNING")
FHIR = settings.fhir_base_url


def _search(resource: str, params: dict) -> list[dict]:
    r = httpx.get(f"{FHIR}/{resource}", params=params, timeout=30)
    r.raise_for_status()
    return [e["resource"] for e in r.json().get("entry", [])]


@mcp.tool()
def get_lab_results(patient_ref: str, test_name: str, max_results: int = 5) -> str:
    """Most recent laboratory results for a patient matching a test name.

    Args:
        patient_ref: FHIR Patient id, for example "1234"
        test_name: Words to match in the test name, for example "HbA1c" or "creatinine"
        max_results: How many results to return, newest first
    """
    obs = _search("Observation", {"subject": f"Patient/{patient_ref}",
                                  "category": "laboratory", "_sort": "-date", "_count": 100})
    hits = [o for o in obs if test_name.lower() in o.get("code", {}).get("text", "").lower()]
    rows = [{"id": f"Observation/{o['id']}", "test": o["code"].get("text"),
             "value": o.get("valueQuantity", {}).get("value"),
             "unit": o.get("valueQuantity", {}).get("unit"),
             "date": o.get("effectiveDateTime")} for o in hits[:max_results]]
    return json.dumps(rows) if rows else f"No results for '{test_name}'"


@mcp.tool()
def get_medication_history(patient_ref: str) -> str:
    """Medicines prescribed to the patient, newest first (shows first-line treatment trials).

    Args:
        patient_ref: FHIR Patient id
    """
    meds = _search("MedicationRequest", {"subject": f"Patient/{patient_ref}",
                                         "_sort": "-authoredon", "_count": 50})
    return json.dumps([{"id": f"MedicationRequest/{m['id']}",
                        "medicine": m.get("medicationCodeableConcept", {}).get("text"),
                        "date": m.get("authoredOn"), "status": m.get("status")}
                       for m in meds])


@mcp.tool()
def retrieve_criteria(specialty: str, query: str) -> str:
    """Referral acceptance criteria most relevant to a query, for one specialty.

    Args:
        specialty: Lower-case specialty, for example "orthopaedics"
        query: Short clinical description of the case
    """
    return format_hits(retrieve(specialty, query))


@mcp.tool()
def log_decline_reason(case_id: str, specialty: str, reason_code: str, detail: str) -> str:
    """Record a predicted decline reason for later analysis of unmet need.

    Args:
        case_id: Case identifier
        specialty: Specialty the referral was for
        reason_code: One of MISSING_INVESTIGATION, BELOW_THRESHOLD, NO_FIRST_LINE, OTHER
        detail: One-sentence explanation (no patient identifiers)
    """
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("INSERT INTO decline_reasons (case_id, specialty, reason_code, detail)"
                     " VALUES (%s, %s, %s, %s)", (case_id, specialty, reason_code, detail))
    return "logged"


# add get_imaging_reports (ImagingStudy / DiagnosticReport) the same way


if __name__ == "__main__":
    mcp.run(transport="streamable-http")