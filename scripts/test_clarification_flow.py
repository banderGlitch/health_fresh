"""
Test clarification flow: POST /analyze -> POST /analyze/continue.

Usage:
  1. Start server: run_server.bat or run_server.ps1 (port 8002)
  2. Run: python scripts/test_clarification_flow.py [--all]
  --all : include tough cases (multi-symptom, ambiguous, jargon)
"""
import argparse
import sys

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

BASE = "http://localhost:8002"  # Match run_server.bat / run_server.ps1

# Test cases: (conversation, demographics, answers, label)
BASIC_CASES = [
    (
        "I have fever and headache",
        {"age": 35, "gender": "male"},
        "3 days, mild, no other symptoms",
        "basic",
    ),
]

TOUGH_CASES = [
    # 1. Multiple symptoms, different durations
    (
        "I have fever, headache, and a cough. Also some stomach discomfort.",
        {"age": 42, "gender": "female"},
        "The fever and headache started 3 days ago. Cough has been there for about 2 weeks. Stomach pain is mild, started yesterday.",
        "multi-symptom diff duration",
    ),
    # 2. Ambiguous short answers
    (
        "Chest tightness and shortness of breath",
        {"age": 58, "gender": "male"},
        "yes, 2 days, severe",
        "ambiguous short",
    ),
    # 3. Medical jargon + lay mix
    (
        "Experiencing dyspnea and diaphoresis",
        {"age": 55, "gender": "female"},
        "Started this morning. Very bad. No chest pain but feel weak.",
        "jargon + lay",
    ),
    # 4. Contradictory / partial
    (
        "High fever and persistent cough",
        {"age": 8, "gender": "male"},
        "Fever for 5 days. Cough I'm not sure, maybe 3 days? Moderate.",
        "partial answers",
    ),
    # 5. Long-form narrative
    (
        "Abdominal pain and vomiting",
        {"age": 35, "gender": "female"},
        "The stomach pain has been severe since yesterday evening. I've vomited 4 times. No fever. Haven't been able to keep food down.",
        "long narrative",
    ),
    # 6. Negative / exclusionary
    (
        "Headache and dizziness",
        {"age": 50, "gender": "male"},
        "Headache for a week, moderate. No chest pain. No shortness of breath. Dizzy on and off.",
        "negations",
    ),
    # 7. Pediatric, mixed severity
    (
        "My 6 year old has fever, cough, and runny nose",
        {"age": 6, "gender": "female"},
        "High fever for 4 days. Cough is mild, maybe 5 days. Runny nose for a week. She's drinking okay, no rash.",
        "pediatric mixed",
    ),
    # 8. Messy informal
    (
        "Chest pain, sweating, feeling breathless",
        {"age": 62, "gender": "male"},
        "umm like 1-2 days? pretty severe. yes sweating a lot. breathlessness worse when i move",
        "messy informal",
    ),
]


def run_case(conversation: str, demographics: dict, answers: str, label: str) -> bool:
    """Run one test case. Returns True if success."""
    try:
        r1 = requests.post(
            f"{BASE}/analyze",
            json={"conversation": conversation, "demographics": demographics},
            timeout=90,
        )
        r1.raise_for_status()
        data = r1.json()
        session_id = data["session_id"]
        questions = data.get("llm_clarification", {}).get("clarifying_questions", [])

        r2 = requests.post(
            f"{BASE}/analyze/continue",
            json={"session_id": session_id, "answers": answers},
            timeout=90,
        )
        r2.raise_for_status()
        data2 = r2.json()

        print(f"  [{label}] triage={data2.get('triage_recommendation')} severity={data2.get('severity')} (before: {data.get('triage_recommendation')})")
        return True
    except Exception as e:
        print(f"  [{label}] FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all cases including tough ones")
    args = parser.parse_args()

    cases = BASIC_CASES + (TOUGH_CASES if args.all else [])
    print("=" * 70)
    print("Clarification flow test (watch API terminal for [CLARIFICATION] logs)")
    print("=" * 70)
    print(f"Running {len(cases)} case(s)...\n")

    ok = 0
    for conv, demo, ans, label in cases:
        if run_case(conv, demo, ans, label):
            ok += 1

    print(f"\n{ok}/{len(cases)} passed. Check API logs for [CLARIFICATION] merged output.")


if __name__ == "__main__":
    try:
        requests.get(f"{BASE}/health", timeout=5)
    except Exception:
        print(f"API not running at {BASE}. Start with: uvicorn api.main:app --reload")
        sys.exit(1)
    main()
