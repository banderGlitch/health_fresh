"""Compare frontend symptom strings to SYMPTOM_LEXICON (run from repo root)."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extraction.symptom_lexicon import SYMPTOM_LEXICON

allowed = set()
for c, vars_ in SYMPTOM_LEXICON.items():
    allowed.add(c.lower())
    for v in vars_:
        allowed.add(v.lower())

text = Path("frontend/src/symptoms.js").read_text(encoding="utf-8")
fe = []
skip = {
    "General",
    "Respiratory",
    "Abdominal",
    "Chest & Heart",
    "Chest & heart",
    "Neurological & senses",
    "Skin",
    "Eyes & Vision",
    "Urinary",
    "Other",
    "SYMPTOMS_BY_CATEGORY",
    "Mild",
    "Moderate",
    "Severe",
}
for m in re.finditer(r'"([^"]+)"', text):
    s = m.group(1)
    if s in skip or s.startswith("Less ") or "days" in s or s.startswith("3") or s.startswith("More "):
        continue
    fe.append(s)

ok = [x for x in fe if x.lower() in allowed]
bad = [x for x in fe if x.lower() not in allowed]
print("IN_LEXICON:", len(ok))
print(ok)
print("NOT_IN_LEXICON:", len(bad))
print(bad)
