"""Generate synthetic GP referral notes and gold labels from your criteria files and HAPI patients.

  uv run python scripts/make_cases.py --preview    # one case per scenario, to read first
  uv run python scripts/make_cases.py --n 80       # the full set (safe to stop and re-run)
"""
import argparse
import json
import random
import re
import sys
import time
from datetime import date
from pathlib import Path

import httpx
from langchain_ollama import ChatOllama
from load_fhir import EXTRA_FAMILY, EXTRA_GIVEN   # the Māori and Pasifika name lists you loaded

from referral_triage.config import settings
from referral_triage.privacy.nhi import is_valid_nhi

SHARES = {"clear_accept": .20, "missing_investigation": .20, "below_threshold": .20,
          "no_first_line": .15, "hidden_red_flag": .15, "ambiguous": .10}
AGE_RANGE = {"orthopaedics": (40, 85)}            # default is (18, 90)
STOP = {"after", "which", "where", "there", "their", "about", "other", "these", "those",
        "whose", "instead", "pathway", "acute", "with", "than", "that"}
OUT = Path("data/test_cases")

COMMON = """Write a realistic referral letter from a New Zealand GP to the {specialty} service.
Patient: {full} (NHI {nhi}), {age}-year-old {sex}, date of birth {dob}.
Contact: phone {phone}; address {address}.
Presenting problem: {problem}.
Style rules:
- Plain prose, 120 to 220 words, one letter, no bullet points and no headings.
- Near the top, state the patient's full name, NHI, date of birth, phone and address exactly as given above.
- Afterwards call the patient {honorific} {family}.
- Never mention guidelines, criteria or checklists.
- Use New Zealand spelling and units (mmol/L, kg, mmHg).
- Output only the letter.
"""

SITUATION = {
    "clear_accept": "Situation: the referral clearly meets every one of these acceptance criteria: {accept}. "
                    "Say that each of the following has been done, with realistic dates in the last 12 months: {required}.",
    "missing_investigation": "Situation: the referral meets these acceptance criteria: {accept}. "
                    "Say that these were done, with realistic dates: {done}. "
                    "Do NOT mention, imply or promise this anywhere in the letter: {gap}.",
    "no_first_line": "Situation: the symptoms have lasted well over 3 months and the letter is otherwise complete. "
                    "Say that these were done, with realistic dates: {done}. "
                    "Do NOT mention that any of this was tried or is planned: {gap}.",
    "below_threshold": "Situation: the case does NOT meet this acceptance criterion: {gap}. "
                    "Make the problem clearly short of that threshold (for example shorter, milder, little impact). "
                    "Investigations and first-line management are otherwise in order.",
    "hidden_red_flag": "Situation: the letter reads like a routine referral. Somewhere in the middle, in one passing clause "
                    "with no emphasis, mention that the patient also has this: {red_flag}. "
                    "Do not use the words 'urgent' or 'red flag' and do not say it is serious.",
    "ambiguous": "Situation: write a vague, hurried letter. Leave unclear how long the symptoms have lasted and whether "
                 "any investigations or first-line treatments were done. Mention no alarming symptoms.",
}


def parse_criteria(path: Path):
    text = path.read_text(encoding="utf-8")
    _, front, body = text.split("---", 2)
    meta = dict(line.split(": ", 1) for line in front.strip().splitlines() if ": " in line)
    crits, cur, sec = {}, None, None
    for line in body.splitlines():
        line = line.rstrip()
        m = re.match(r"^## (\S+)\s+(.*)$", line)
        if m:
            cur = crits[m.group(1)] = {"problem": m.group(2), "accept": [], "required": [],
                                       "red_flags": [], "alternatives": []}
            sec = None
            continue
        h = re.match(r"^### (.*)$", line)
        if cur is not None and h:
            t = h.group(1).lower()
            sec = ("accept" if t.startswith("accept") else "required" if t.startswith("required")
                   else "red_flags" if t.startswith("red flag")
                   else "alternatives" if t.startswith("alternative") else None)
            continue
        if cur is not None and sec and line.startswith("- "):
            cur[sec].append(line[2:].strip())
    complete = {k: v for k, v in crits.items() if all(v[s] for s in ("accept", "required", "red_flags", "alternatives"))}
    for k in set(crits) - set(complete):
        print(f"  skipping {k}: it needs all four sections with at least one bullet each")
    return meta["specialty"], complete


def to_patient(r: dict):
    if r.get("deceasedDateTime") or r.get("deceasedBoolean"):
        return None
    try:
        nhi = r["identifier"][0]["value"]
        given, family = r["name"][0]["given"][0], r["name"][0]["family"]
        born = date.fromisoformat(r["birthDate"])
        a = r["address"][0]
        phone = r["telecom"][0]["value"]
    except (KeyError, IndexError, ValueError):
        return None
    if not is_valid_nhi(nhi):
        return None
    t = date.today()
    age = t.year - born.year - ((t.month, t.day) < (born.month, born.day))
    sex = r.get("gender", "unknown")
    return {"ref": r["id"], "nhi": nhi, "given": given, "family": family, "full": f"{given} {family}",
            "age": age, "sex": sex, "dob": born.strftime("%d/%m/%Y"), "phone": phone,
            "address": f"{a['line'][0]}, {a['city']} {a.get('postalCode', '')}".strip(),
            "honorific": "Mr" if sex == "male" else "Ms",
            "name_group": "maori_pasifika_name" if given in EXTRA_GIVEN or family in EXTRA_FAMILY else "other_name"}


def fetch_patients():
    url, params, out = f"{settings.fhir_base_url}/Patient", {"_count": 100}, []
    while url:
        r = httpx.get(url, params=params, timeout=60)
        r.raise_for_status()
        body = r.json()
        out += [p for e in body.get("entry", []) if (p := to_patient(e["resource"]))]
        url = next((link["url"] for link in body.get("link", []) if link["relation"] == "next"), None)
        params = None
    return sorted(out, key=lambda p: int(p["ref"]) if p["ref"].isdigit() else p["ref"])


def scenario_list(n: int):
    counts = {k: int(n * v) for k, v in SHARES.items()}
    for k in SHARES:                                   # hand out the rounding remainder
        if sum(counts.values()) >= n:
            break
        counts[k] += 1
    return [k for k, c in counts.items() for _ in range(c)]


def build_plan(n, specs, patients, seed=42):
    rng, plan, names = random.Random(seed), [], sorted(specs)
    for si, spec in enumerate(names):
        n_s = n // len(names) + (1 if si < n % len(names) else 0)
        lo, hi = AGE_RANGE.get(spec, (18, 90))
        pool = [p for p in patients if lo <= p["age"] <= hi]
        rng.shuffle(pool)
        scns = scenario_list(n_s)
        rng.shuffle(scns)
        if len(pool) < len(scns):
            print(f"  only {len(pool)} suitable patients for {spec}; making {len(pool)} cases, not {len(scns)}")
            scns = scns[:len(pool)]
        for i, (scn, p) in enumerate(zip(scns, pool), 1):
            plan.append({"case_id": f"{spec[:4].upper()}-{i:03d}", "specialty": spec, "scenario": scn,
                         "patient": p, "criterion_id": rng.choice(sorted(specs[spec])),
                         "seed": rng.randint(1, 10**6)})
    return plan


def prompt_and_gold(item, crit, cid):
    p, scn = item["patient"], item["scenario"]
    req = crit["required"]
    trial = [r for r in req if re.search(r"\btrial|programme|program\b", r, re.I)]
    invest = [r for r in req if r not in trial]
    gap = {"missing_investigation": (invest or req)[0],
           "no_first_line": (trial or req[-1:])[0],
           "below_threshold": crit["accept"][0]}.get(scn, "")
    done = [r for r in req if r != gap]
    red = random.Random(item["seed"]).choice(crit["red_flags"])
    text = COMMON.format(specialty=item["specialty"], problem=crit["problem"], **p) + SITUATION[scn].format(
        accept="; ".join(crit["accept"]), required="; ".join(req), done="; ".join(done) or "all requirements",
        gap=gap, red_flag=red)
    gaps = ([gap] if scn in ("missing_investigation", "no_first_line")
            else [f"Acceptance criterion not met: {gap}"] if scn == "below_threshold" else [])
    decision = {"clear_accept": "READY_TO_SEND", "missing_investigation": "MISSING_INFO",
                "no_first_line": "MISSING_INFO", "below_threshold": "CONSIDER_ALTERNATIVE",
                "hidden_red_flag": "URGENT_ESCALATE", "ambiguous": "MISSING_INFO"}[scn]
    return text, {"decision": decision, "red_flag": scn == "hidden_red_flag", "required_gaps": gaps,
                  "criterion_ids": [cid]}, red


def words(s):
    return set(re.findall(r"[a-z]{5,}", s.lower())) - STOP


def problems(note, p, scn, crit, red):
    out, low = [], note.lower()
    if p["nhi"] not in note:
        out.append("NHI missing")
    if p["full"] not in note:
        out.append("full name missing")
    if scn == "hidden_red_flag":
        if not any(w in low for w in words(red)):
            out.append("red flag not mentioned")
    else:
        ordinary = words(crit["problem"] + " " + " ".join(crit["accept"] + crit["required"]))
        hit = sorted(w for w in words(" ".join(crit["red_flags"])) - ordinary if w in low)
        if hit:
            out.append("possible red-flag wording: " + ", ".join(hit))
    n = len(note.split())
    if not 80 <= n <= 320:
        out.append(f"length {n} words")
    return out


def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return re.sub(r"^```\w*\n|\n```$", "", text).strip()


def make_llm(seed):
    return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url,
                      temperature=0.8, num_ctx=8192, reasoning=False, seed=seed)


HARD = ("NHI missing", "full name missing", "red flag not mentioned")


def generate(prompt, item, crit, red):
    note, probs = "", []
    for attempt in range(2):
        extra = "" if attempt == 0 else "\nIMPORTANT: your last attempt broke these rules: " + "; ".join(probs)
        note = clean(make_llm(item["seed"] + attempt).invoke(prompt + extra).content)
        probs = problems(note, item["patient"], item["scenario"], crit, red)
        if not any(x.startswith(HARD) for x in probs):
            break
    return note, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    specs = {}
    for path in sorted(Path("data/criteria").glob("*.md")):
        spec, crits = parse_criteria(path)
        if crits:
            specs[spec] = crits
    if not specs:
        sys.exit("No usable criteria found in data/criteria/")
    patients = fetch_patients()
    print(f"{len(patients)} living adult patients with valid NHIs; specialties: {', '.join(specs)}")
    if not patients:
        sys.exit("No patients found. Is HAPI up, and did `make data` finish?")

    plan = build_plan(args.n, specs, patients)
    if args.preview:
        seen, first = set(), []
        for it in plan:
            if it["scenario"] not in seen:
                seen.add(it["scenario"])
                first.append(it)
        plan = [dict(it, case_id="PREVIEW-" + it["case_id"]) for it in first]
    target = OUT / ("preview.json" if args.preview else "gold.json")
    OUT.mkdir(parents=True, exist_ok=True)
    cases = json.loads(target.read_text()) if target.exists() else []
    have = {c["case_id"] for c in cases}
    todo = [it for it in plan if it["case_id"] not in have]
    print(f"{len(have)} already done, {len(todo)} to generate -> {target}")

    for i, item in enumerate(todo, 1):
        t0, p = time.time(), item["patient"]
        crit = specs[item["specialty"]][item["criterion_id"]]
        prompt, gold, red = prompt_and_gold(item, crit, item["criterion_id"])
        note, probs = generate(prompt, item, crit, red)
        cases.append({
            "case_id": item["case_id"], "specialty": item["specialty"], "patient_ref": p["ref"],
            "nhi": p["nhi"], "patient_names": [p["full"], p["family"]], "scenario": item["scenario"],
            "ambiguous": item["scenario"] == "ambiguous", "referral_note": note, "gold": gold,
            "fairness": {"sex": p["sex"], "age_band": f"{p['age'] // 10 * 10}-{p['age'] // 10 * 10 + 9}",
                         "name_group": p["name_group"]},
            "review": {"gp_reviewed": False, "gp_label": None, "needs_check": probs, "my_check": None}})
        target.write_text(json.dumps(cases, indent=2, ensure_ascii=False))     # saved after every case
        flag = f"  CHECK: {'; '.join(probs)}" if probs else ""
        print(f"[{i}/{len(todo)}] {item['case_id']} {item['scenario']} ({time.time() - t0:.0f}s){flag}")
    print(f"done: {len(cases)} cases in {target}")


if __name__ == "__main__":
    main()
