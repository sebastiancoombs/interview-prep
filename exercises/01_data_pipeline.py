"""Exercise 1: repair this JSONL + CSV processing pipeline."""

import csv
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"


def load_events(path: Path) -> list[dict]:
    with path.open() as handle:
        return json.load(handle)


def load_labels(path: Path) -> dict[int, dict]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            int(row["user_id"]): {
                "label": int(row["label"]),
                "weight": float(row["weight"]),
            }
            for row in rows
        }


def build_rows(events: list[dict], labels: dict[int, dict]) -> list[dict]:
    output = []
    for event in events:
        label = labels[event["user_id"]]
        output.append(
            {
                "user_id": event["user_id"],
                "token_count": len(event["tokens"]),
                **label,
            }
        )
    return output


def main() -> None:
    events = load_events(DATA_DIR / "events.jsonl")
    labels = load_labels(DATA_DIR / "labels.csv")
    rows = build_rows(events, labels)

    output_path = DATA_DIR / "processed.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
