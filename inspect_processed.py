import numpy as np
from pathlib import Path

root = Path("./processed/anorthite_compression/samples")
files = sorted(root.glob("sample_*.npz"))

total = []
affine = []
nonaffine = []
recon = []

for f in files:
    p = np.load(f)

    total.append(
        np.sqrt(np.mean(np.sum(p["total_displacement"] ** 2, axis=1)))
    )
    affine.append(
        np.sqrt(np.mean(np.sum(p["affine_displacement"] ** 2, axis=1)))
    )
    nonaffine.append(
        np.sqrt(np.mean(np.sum(p["nonaffine_displacement"] ** 2, axis=1)))
    )
    recon.append(float(p["reconstruction_max_error"]))

total = np.array(total)
affine = np.array(affine)
nonaffine = np.array(nonaffine)
recon = np.array(recon)

print(f"Transitions: {len(files)}")
print()

print("TOTAL DISPLACEMENT RMS")
print(f"  min:  {total.min():.6f} Å")
print(f"  mean: {total.mean():.6f} Å")
print(f"  max:  {total.max():.6f} Å")
print()

print("AFFINE DISPLACEMENT RMS")
print(f"  min:  {affine.min():.9f} Å")
print(f"  mean: {affine.mean():.9f} Å")
print(f"  max:  {affine.max():.9f} Å")
print()

print("NON-AFFINE DISPLACEMENT RMS")
print(f"  min:  {nonaffine.min():.6f} Å")
print(f"  mean: {nonaffine.mean():.6f} Å")
print(f"  max:  {nonaffine.max():.6f} Å")
print()

print("RECONSTRUCTION ERROR")
print(f"  max: {recon.max():.3e} Å")