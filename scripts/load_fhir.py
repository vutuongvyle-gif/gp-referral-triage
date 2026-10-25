"""NZ-ise Synthea bundles (test NHIs, NZ names and addresses) and load them into HAPI."""
import json
import random
from pathlib import Path

import httpx
from faker import Faker

from referral_triage.config import settings
from referral_triage.privacy.nhi import make_test_nhi

NHI_SYSTEM = "https://standards.digital.health.nz/ns/nhi-id"
SRC = Path("data/synthea/fhir")
fake, rng = Faker("en_NZ"), random.Random(42)
Faker.seed(42)
# Add Māori and Pasifika given/family names here so PII recall is tested fairly
EXTRA_GIVEN = ["Aroha", "Mere", "Tama", "Wiremu", "Losa", "Sione"]
EXTRA_FAMILY = ["Ngata", "Parata", "Tuhiwai", "Fa'asalele", "Tupou"]


def nz_ise(bundle: dict) -> dict:
    for entry in bundle["entry"]:
        r = entry["resource"]
        if r["resourceType"] != "Patient":
            continue
        given = rng.choice(EXTRA_GIVEN) if rng.random() < 0.3 else (
            fake.first_name_female() if r.get("gender") == "female" else fake.first_name_male())
        family = rng.choice(EXTRA_FAMILY) if rng.random() < 0.3 else fake.last_name()
        r["identifier"] = [{"system": NHI_SYSTEM, "value": make_test_nhi(rng)}]
        r["name"] = [{"use": "official", "family": family, "given": [given]}]
        r["address"] = [{"line": [fake.street_address()], "city": fake.city(),
                         "postalCode": fake.postcode(), "country": "NZ"}]
        r["telecom"] = [{"system": "phone", "value": fake.phone_number()}]
    return bundle


def post(bundle: dict) -> None:
    r = httpx.post(settings.fhir_base_url, json=bundle, timeout=600,
                   headers={"Content-Type": "application/fhir+json"})
    r.raise_for_status()


def main():
    files = sorted(SRC.glob("*.json"))
    infra = [f for f in files if f.name.startswith(("hospital", "practitioner"))]
    for f in infra:                                   # organisations first
        post(json.loads(f.read_text()))
    patients = [f for f in files if f not in infra]
    for i, f in enumerate(patients, 1):
        post(nz_ise(json.loads(f.read_text())))
        print(f"loaded {i}/{len(patients)}", end="\r")
    print("\ndone")


if __name__ == "__main__":
    main()