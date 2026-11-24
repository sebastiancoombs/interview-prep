import random


def split_records(records: list[dict], train_fraction: float) -> tuple[list[dict], list[dict]]:
    shuffled = records.copy()
    random.Random(7).shuffle(shuffled)
    split_at = int(len(shuffled) * train_fraction)
    return shuffled[:split_at], shuffled[split_at:]


records = [{"id": item_id, "value": item_id * 10} for item_id in range(10)]
train, validation = split_records(records, 0.8)
train_ids = {row["id"] for row in train}
validation_ids = {row["id"] for row in validation}
assert len(train) == 8 and len(validation) == 2
assert train_ids.isdisjoint(validation_ids)
print("train IDs:", sorted(train_ids))
print("validation IDs:", sorted(validation_ids))
