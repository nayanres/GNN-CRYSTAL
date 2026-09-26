import os
import json
import shutil
import zipfile

ROOT = r"C:\Users\games\Downloads\Diopside_Anorthite_MatsuiPot\Diopside_Anorthite_MatsuiPot\processed_production"

OUT_DIR = os.path.join(os.path.dirname(ROOT), "abc_train_val")
ZIP_PATH = os.path.join(os.path.dirname(ROOT), "abc_train_val.zip")

# 
# Read official split manifest
# 

with open(os.path.join(ROOT, "splits.json"), "r") as f:
    split_info = json.load(f)

train_runs = split_info["splits"]["train"]
val_runs = split_info["splits"]["val"]

selected = train_runs + val_runs

print("Train trajectories:", len(train_runs))
print("Val trajectories:", len(val_runs))
print("Total selected:", len(selected))

assert len(train_runs) == 12
assert len(val_runs) == 12
assert len(selected) == 24

# 
# Start clean
# 

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)

os.makedirs(OUT_DIR)

# Keep split manifest so training code works normally.
shutil.copy2(
    os.path.join(ROOT, "splits.json"),
    os.path.join(OUT_DIR, "splits.json")
)

# 
# Copy ONLY train + validation trajectories
# 

total_samples = 0

for i, entry in enumerate(selected, 1):

    run_id = entry["run_id"]

    src = os.path.join(ROOT, run_id)
    dst = os.path.join(OUT_DIR, run_id)

    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"Missing trajectory directory: {src}"
        )

    shutil.copytree(src, dst)

    sample_dir = os.path.join(dst, "samples")

    samples = [
        f for f in os.listdir(sample_dir)
        if f.startswith("sample_") and f.endswith(".npz")
    ]

    expected = entry["num_samples"]

    if len(samples) != expected:
        raise RuntimeError(
            f"{run_id}: expected {expected} samples, "
            f"found {len(samples)}"
        )

    total_samples += len(samples)

    print(
        f"[{i:02d}/24] {run_id}: "
        f"{len(samples)} samples"
    )

# 
# Critical checks
# 

assert total_samples == 12_000

# Make sure NO test trajectory accidentally entered archive.
test_ids = {
    entry["run_id"]
    for entry in split_info["splits"]["test"]
}

copied_dirs = {
    name for name in os.listdir(OUT_DIR)
    if os.path.isdir(os.path.join(OUT_DIR, name))
}

leaked = test_ids.intersection(copied_dirs)

if leaked:
    raise RuntimeError(
        f"TEST DATA LEAKED INTO ARCHIVE: {sorted(leaked)}"
    )

print()
print("Verified:")
print("  Train samples: 6000")
print("  Val samples:   6000")
print("  Total samples: 12000")
print("  Test trajectories copied: 0")

# 
# ZIP
# 

print()
print("Creating ZIP...")

with zipfile.ZipFile(
    ZIP_PATH,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6
) as zf:

    for current_root, dirs, files in os.walk(OUT_DIR):

        for filename in files:

            full_path = os.path.join(
                current_root,
                filename
            )

            # Archive root becomes processed_production/
            relative = os.path.relpath(
                full_path,
                OUT_DIR
            )

            archive_name = os.path.join(
                "processed_production",
                relative
            )

            zf.write(
                full_path,
                archive_name
            )

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print("ZIP:", ZIP_PATH)
print(
    "Size:",
    round(os.path.getsize(ZIP_PATH) / 1024**3, 3),
    "GB"
)
print()
print("TEST SET WAS NOT INCLUDED.")