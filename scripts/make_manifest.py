import json
from pathlib import Path
from datetime import datetime

REPORTS_DIR = Path("artifacts/inference/val/reports")  # change split if needed
OUT_MANIFEST = Path("artifacts/inference/val/manifest.jsonl")
OUT_PATIENTS = Path("artifacts/inference/val/patients_seed.json")

FIRST_NAMES = [
    "john", "jane", "michael", "emma", "david", "olivia",
    "daniel", "sophia", "alex", "mia", "noah", "ava",
    "liam", "isabella", "ryan", "charlotte"
]

LAST_NAMES = [
    "doe", "smith", "wilson", "brown", "khan", "lee",
    "garcia", "martin", "clark", "hall", "young", "king"
]

def generate_name(index):
    first = FIRST_NAMES[index % len(FIRST_NAMES)]
    last = LAST_NAMES[index % len(LAST_NAMES)]
    return first, last

def main():
    report_files = sorted(REPORTS_DIR.glob("*.json"))
    if not report_files:
        raise FileNotFoundError(f"No report JSON files found in {REPORTS_DIR}")

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")

    patients = []
    manifest_lines = []

    for i, rf in enumerate(report_files, start=1):
        report = json.loads(rf.read_text(encoding="utf-8"))
        image = report["image"]

        first, last = generate_name(i)
        patient_id = f"p_{i:03d}_{first}_{last}"
        xray_id = f"XRAY#{patient_id}#{today}#0001"

        # Manifest entry
        manifest_lines.append({
            "image": image,
            "patientId": patient_id,
            "xrayId": xray_id
        })

        # Patient seed entry
        patients.append({
            "patientId": patient_id,
            "name": f"{first.title()} {last.title()}",
            "email": f"{first}.{last}{i}@example.com",
            "phoneE164": f"+1555000{i:04d}"
        })

    # Write manifest
    with OUT_MANIFEST.open("w", encoding="utf-8") as f:
        for rec in manifest_lines:
            f.write(json.dumps(rec) + "\n")

    # Write patients seed file
    with OUT_PATIENTS.open("w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2)

    print(f"Created manifest: {OUT_MANIFEST}")
    print(f"Created patient seed file: {OUT_PATIENTS}")
    print(f"Total patients created: {len(patients)}")

if __name__ == "__main__":
    main()