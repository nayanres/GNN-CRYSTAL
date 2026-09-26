import os
import csv
import json
import glob

RAW_MANIFEST = r".\production_runs\manifest.csv"
PROCESSED = r".\processed_production"
OUTPUT = r".\processed_production\splits.json"

SEED_TO_SPLIT = {
    "11001": "train",
    "22002": "val",
    "33003": "test",
}

with open(RAW_MANIFEST, newline="") as f:
    rows = list(csv.DictReader(f))

splits = {
    "train": [],
    "val": [],
    "test": [],
}

summary = {
    "train": {"trajectories": 0, "samples": 0},
    "val":   {"trajectories": 0, "samples": 0},
    "test":  {"trajectories": 0, "samples": 0},
}

for row in rows:

    run_id = row["run_id"]
    seed = str(row["seed"])

    if seed not in SEED_TO_SPLIT:
        raise ValueError(f"Unknown seed: {seed}")

    split = SEED_TO_SPLIT[seed]

    sample_pattern = os.path.join(
        PROCESSED,
        run_id,
        "samples",
        "sample_*.npz"
    )

    samples = sorted(glob.glob(sample_pattern))

    if len(samples) != 500:
        raise ValueError(
            f"{run_id}: expected 500 samples, got {len(samples)}"
        )

    entry = {
        "run_id": run_id,
        "loading_mode": row["loading_mode"],
        "final_scale": float(row["final_scale"]),
        "seed": int(seed),
        "sample_dir": os.path.join(
            run_id,
            "samples"
        ),
        "num_samples": len(samples),
    }

    splits[split].append(entry)

    summary[split]["trajectories"] += 1
    summary[split]["samples"] += len(samples)


# 
# Sanity checks
# 

all_ids = []

for split in splits:
    all_ids.extend(
        x["run_id"] for x in splits[split]
    )

assert len(all_ids) == 36
assert len(set(all_ids)) == 36

assert summary["train"]["samples"] == 6000
assert summary["val"]["samples"] == 6000
assert summary["test"]["samples"] == 6000


output = {
    "description":
        "Trajectory-level seed split. "
        "All loading conditions occur in all partitions; "
        "thermal initialization is disjoint.",

    "split_rule": {
        "train_seed": 11001,
        "val_seed": 22002,
        "test_seed": 33003,
    },

    "summary": summary,
    "splits": splits,
}


with open(OUTPUT, "w") as f:
    json.dump(output, f, indent=2)


print("=" * 60)
print("TRAJECTORY SPLIT CREATED")
print("=" * 60)

for split in ["train", "val", "test"]:
    print(
        f"{split.upper():5}  "
        f"trajectories={summary[split]['trajectories']:2d}  "
        f"samples={summary[split]['samples']:5d}"
    )

print()
print("Total trajectories:", len(all_ids))
print("Unique trajectories:", len(set(all_ids)))
print("Split manifest:", OUTPUT)