import json
from pathlib import Path

# Adjust this path to where your folds are saved
fold_dir = Path("./splits/kfold_10")
test_sets = []

for f in fold_dir.glob("fold_*.json"):
    data = json.loads(f.read_text())
    test_sets.append(set(data["test"]))

# Check if Fold 0 and Fold 1 have any overlapping test samples
overlap = test_sets[0].intersection(test_sets[1])
print(f"Overlap between Fold 0 and 1 test sets: {len(overlap)} samples")
# This should be 0!