"""Exercise 2: repair the deterministic, disjoint dataset split."""

import random


def split_records(records: list[dict], train_fraction: float) -> tuple[list[dict], list[dict]]:
    random.shuffle(records)
    split_at = int(len(records) * train_fraction)
    train = records[:split_at]
    validation = records[: len(records) - split_at]
    return train, validation


def main() -> None:
    records = [{"id": item_id, "value": item_id * 10} for item_id in range(10)]
    train, validation = split_records(records, train_fraction=0.8)

    train_ids = {row["id"] for row in train}
    validation_ids = {row["id"] for row in validation}
    assert len(train) == 8
    assert len(validation) == 2
    assert train_ids.isdisjoint(validation_ids), "data leakage between splits"
    print("train IDs:", sorted(train_ids))
    print("validation IDs:", sorted(validation_ids))


if __name__ == "__main__":
    main()
