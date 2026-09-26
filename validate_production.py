import csv
import glob
import os
import numpy as np

ROOT = r".\production_runs"
manifest_path = os.path.join(ROOT, "manifest.csv")

# 
# Manifest
# 

with open(manifest_path, newline="") as f:
    manifest = list(csv.DictReader(f))

print("=" * 65)
print("ANORTHITE RAW DATASET VALIDATION")
print("=" * 65)
print("Manifest trajectories:", len(manifest))

assert len(manifest) == 36, f"Expected 36 runs, found {len(manifest)}"

failures = []
total_frames = 0
all_dets = []
all_timesteps = []

for i, row in enumerate(manifest, 1):

    runid = row["run_id"]
    run_dir = os.path.join(ROOT, runid)

    traj = os.path.join(
        run_dir,
        f"traj_{runid}.lammpstrj"
    )

    if not os.path.isfile(traj):
        failures.append((runid, "trajectory missing"))
        continue

    frames = 0
    timesteps = []
    determinants = []

    try:
        with open(traj, "r") as f:

            while True:

                line = f.readline()

                if not line:
                    break

                if not line.startswith("ITEM: TIMESTEP"):
                    continue

                # timestep
                step = int(f.readline().strip())
                timesteps.append(step)

                # NUMBER OF ATOMS
                header = f.readline().strip()
                assert header == "ITEM: NUMBER OF ATOMS"

                natoms = int(f.readline().strip())

                if natoms != 104:
                    raise ValueError(
                        f"expected 104 atoms, got {natoms}"
                    )

                # BOX BOUNDS
                box_header = f.readline().strip()

                if not box_header.startswith("ITEM: BOX BOUNDS"):
                    raise ValueError("missing BOX BOUNDS")

                bounds = [
                    list(map(float, f.readline().split()))
                    for _ in range(3)
                ]

                # Restricted triclinic box
                if all(len(x) == 3 for x in bounds):

                    xlo_b, xhi_b, xy = bounds[0]
                    ylo_b, yhi_b, xz = bounds[1]
                    zlo_b, zhi_b, yz = bounds[2]

                    xlo = xlo_b - min(0.0, xy, xz, xy + xz)
                    xhi = xhi_b - max(0.0, xy, xz, xy + xz)

                    ylo = ylo_b - min(0.0, yz)
                    yhi = yhi_b - max(0.0, yz)

                    lx = xhi - xlo
                    ly = yhi - ylo
                    lz = zhi_b - zlo_b

                    H = np.array([
                        [lx, 0.0, 0.0],
                        [xy, ly, 0.0],
                        [xz, yz, lz]
                    ])

                # Orthogonal fallback
                elif all(len(x) == 2 for x in bounds):

                    lx = bounds[0][1] - bounds[0][0]
                    ly = bounds[1][1] - bounds[1][0]
                    lz = bounds[2][1] - bounds[2][0]

                    H = np.diag([lx, ly, lz])

                else:
                    raise ValueError(
                        "unrecognized BOX BOUNDS format"
                    )

                det = np.linalg.det(H)

                if not np.isfinite(det) or det <= 0:
                    raise ValueError(
                        f"invalid cell determinant {det}"
                    )

                determinants.append(det)

                # ATOMS
                atom_header = f.readline().strip()

                if not atom_header.startswith("ITEM: ATOMS"):
                    raise ValueError("missing ATOMS header")

                columns = atom_header.split()[2:]

                required = {
                    "id", "type", "q",
                    "xu", "yu", "zu",
                    "vx", "vy", "vz",
                    "fx", "fy", "fz"
                }

                if not required.issubset(columns):
                    missing = required - set(columns)
                    raise ValueError(
                        f"missing atom fields: {missing}"
                    )

                ids = []

                for _ in range(natoms):

                    values = f.readline().split()

                    if len(values) != len(columns):
                        raise ValueError(
                            "atom row column mismatch"
                        )

                    numeric = np.asarray(
                        values,
                        dtype=float
                    )

                    if not np.isfinite(numeric).all():
                        raise ValueError(
                            "non-finite atom data"
                        )

                    ids.append(
                        int(values[columns.index("id")])
                    )

                if sorted(ids) != list(range(1, 105)):
                    raise ValueError(
                        "invalid/duplicate atom IDs"
                    )

                frames += 1

        if frames != 501:
            raise ValueError(
                f"expected 501 frames, got {frames}"
            )

        if len(timesteps) != len(set(timesteps)):
            raise ValueError("duplicate timesteps")

        if not all(
            b > a for a, b in zip(timesteps, timesteps[1:])
        ):
            raise ValueError(
                "timesteps not strictly increasing"
            )

        if timesteps[0] != 0 or timesteps[-1] != 5000:
            raise ValueError(
                f"unexpected timestep range "
                f"{timesteps[0]}->{timesteps[-1]}"
            )

        total_frames += frames
        all_dets.extend(determinants)
        all_timesteps.extend(timesteps)

        print(
            f"[{i:02d}/36] PASS  "
            f"{runid:<28} "
            f"frames={frames} "
            f"V={determinants[0]:.3f}"
            f"->{determinants[-1]:.3f}"
        )

    except Exception as e:
        failures.append((runid, str(e)))
        print(
            f"[{i:02d}/36] FAIL  "
            f"{runid}: {e}"
        )


print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)

print("Valid trajectories:", 36 - len(failures))
print("Failed trajectories:", len(failures))
print("Total frames:", total_frames)

if all_dets:
    print(
        "Cell volume range:",
        f"{min(all_dets):.6f} -> {max(all_dets):.6f} A^3"
    )

if failures:
    print("\nFAILURES:")
    for runid, reason in failures:
        print(f"  {runid}: {reason}")
    raise SystemExit(1)

print()
print("RAW DATASET VALIDATION PASSED")
print("Expected transitions:", 36 * 500)
