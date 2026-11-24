import csv
import json
from pathlib import Path


DATA_DIR = Path(__file__).parents[1] / "exercises" / "data"


def load_events(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_labels(path: Path) -> dict[str, dict]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            row["user_id"]: {
                "label": int(row["label"]),
                "weight": float(row["weight"]) if row["weight"] else None,
            }
            for row in rows
        }


events = load_events(DATA_DIR / "events.jsonl")
labels = load_labels(DATA_DIR / "labels.csv")
rows = [
    {
        "user_id": event["user_id"],
        "token_count": len(event["tokens"]),
        **labels[event["user_id"]],
    }
    for event in events
]

output_path = DATA_DIR / "processed.csv"
with output_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {output_path}")
