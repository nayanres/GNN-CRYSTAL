# 
# ANORTHITE DYNAMICS — FULL FIXED EXPERIMENT
#
# Purpose
# -------
# 1) E(3)/O(3)-equivariant A/B/C state ablation
#       A = species
#       B = species + velocity
#       C = species + velocity + force
#
# 2) Strong non-learned physics baselines
#       zero
#       ballistic:      v_t * dt
#       Taylor/Verlet:  v_t * dt + 0.5 * a_t * dt^2
#
# 3) Strong node-only O(3)-equivariant baselines
#       B = learned scalar(v invariants, species) * v
#       C = learned scalar coefficients on v and f
#
# 4) Matched-preprocessing NNConv-C baseline
#       same categorical species representation
#       same isotropic vector scaling
#       no vector mean subtraction
#       same train/val trajectories
#
# 5) Numerical equivariance gates before E3 training
#       proper SO(3) rotation
#       improper O(3) reflection
#
# 6) Post-training symmetry stress tests
#       10 fixed random SO(3) rotations
#       3 fixed improper O(3) transforms
#
# 7) Detailed validation reporting
#       component RMSE / MAE
#       vector error mean / RMS / median / p95 / max
#       trajectory-level RMSE mean/std and each trajectory
#       per-species metrics
#
# TEST remains physically absent and is never loaded/evaluated.
#
# Designed for Colab + Tesla T4-class GPU.
# Standard Python: can be pasted into one Colab cell or saved as .py.
# 


# 
# Dependency bootstrap
# 

import importlib
import importlib.util
import subprocess
import sys


def _ensure_package(import_name, pip_spec, required_version=None):
    needs_install = importlib.util.find_spec(import_name) is None

    if not needs_install and required_version is not None:
        module = importlib.import_module(import_name)
        current = getattr(module, "__version__", None)
        needs_install = current != required_version

    if needs_install:
        print(f"Installing {pip_spec} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_spec]
        )
        importlib.invalidate_caches()


_ensure_package("e3nn", "e3nn==0.6.0", required_version="0.6.0")
_ensure_package("torch_geometric", "torch-geometric")


# 
# Imports
# 

import os
import json
import math
import random
import time
import shutil
import platform
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv

import e3nn
from e3nn import o3


# 
# Configuration
# 

ROOT = "/content/processed_production"
SPLIT_FILE = os.path.join(ROOT, "splits.json")

OUTPUT_DIR = "/content/anorthite_full_fixed_outputs"
CLEAN_OUTPUT_DIR = True

# One seed is appropriate for the current grant-stage run.
# For paper replication later:
# TRAIN_SEEDS = [318, 2718, 31415]
TRAIN_SEEDS = [318]

ROTATION_SEED = 8675309

BATCH_SIZE = 48
MAX_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

VAL_EVERY = 5
PATIENCE_EVALS = 6
GRAD_CLIP_NORM = 10.0

CUTOFF = 3.5

# E3 model
SPECIES_EMBED_DIM = 16
HIDDEN_IRREPS = o3.Irreps("32x0e + 16x1o")
EDGE_IRREPS = o3.Irreps.spherical_harmonics(lmax=2)
OUTPUT_IRREPS = o3.Irreps("1x1o")

NUM_RADIAL = 8
RADIAL_HIDDEN = 64

# Matched NNConv
NN_HIDDEN = 64

# Node-only equivariant baseline
NODE_BASELINE_HIDDEN = 64

# Stress tests
N_SO3_ROTATIONS = 10
N_RANDOM_IMPROPER = 1

# Fail-fast numerical equivariance gate.
# FP32 e3nn should normally be far below this.
EQ_SELFTEST_RMSE_TOL = 5e-5
EQ_SELFTEST_MAX_TOL = 2e-4
SELFTEST_GRAPHS = 8

# LAMMPS "metal" units.
# Production script used timestep 0.001 ps and dump every 10 MD steps.
MD_TIMESTEP_PS = 0.001
EXPECTED_FRAME_STRIDE = 10

# Matsui type mapping used in the source LAMMPS setup:
# 1 Ca, 2 Mg, 3 Al, 4 Si, 5 O
ATOMIC_NUMBER_BY_RAW_TYPE = {
    1: 20,  # Ca
    2: 12,  # Mg
    3: 13,  # Al
    4: 14,  # Si
    5: 8,   # O
}

ELEMENT_BY_Z = {
    8: "O",
    12: "Mg",
    13: "Al",
    14: "Si",
    20: "Ca",
}

MASS_AMU_BY_Z = {
    8: 15.999,
    12: 24.305,
    13: 26.9815385,
    14: 28.085,
    20: 40.078,
}

MAX_ATOMIC_NUMBER = 118

# Run switches
RUN_PHYSICS_BASELINES = True
RUN_NODE_EQUIV_BASELINES = True
RUN_E3_ABC = True
RUN_MATCHED_NNCONV_C = True
RUN_SYMMETRY_STRESS_TESTS = True

# AMP is useful for conventional baselines but intentionally disabled for E3.
USE_AMP_FOR_NON_E3 = torch.cuda.is_available()


# 
# Output directory
# 

if CLEAN_OUTPUT_DIR and os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 
# Device / environment
# 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print("=" * 80)
print("ANORTHITE FULL FIXED EXPERIMENT")
print("=" * 80)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("e3nn:", e3nn.__version__)
print("torch-geometric:", importlib.import_module("torch_geometric").__version__)
print("E3 hidden irreps:", HIDDEN_IRREPS)
print("Edge irreps:", EDGE_IRREPS)
print("Output irreps:", OUTPUT_IRREPS)
print("E3 AMP: disabled")
print("Non-E3 AMP:", USE_AMP_FOR_NON_E3)
print()


# 
# Reproducibility helpers
# 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(TRAIN_SEEDS[0])


# 
# Split manifest
# 

with open(SPLIT_FILE, "r") as f:
    split_info = json.load(f)

train_runs = split_info["splits"]["train"]
val_runs = split_info["splits"]["val"]
test_runs = split_info["splits"]["test"]

assert len(train_runs) == 12, len(train_runs)
assert len(val_runs) == 12, len(val_runs)
assert len(test_runs) == 12, len(test_runs)

train_run_ids = [x["run_id"] for x in train_runs]
val_run_ids = [x["run_id"] for x in val_runs]
test_run_ids = [x["run_id"] for x in test_runs]

assert set(train_run_ids).isdisjoint(val_run_ids)
assert set(train_run_ids).isdisjoint(test_run_ids)
assert set(val_run_ids).isdisjoint(test_run_ids)

print("Train trajectories:", len(train_runs))
print("Val trajectories:  ", len(val_runs))
print("Test trajectories: ", len(test_runs), "(manifest only)")


# 
# Verify TEST is physically absent
# 

test_present = []

for entry in test_runs:
    candidates = [
        os.path.join(ROOT, entry["run_id"]),
        os.path.join(
            ROOT,
            entry["sample_dir"].replace("\\", "/")
        ),
    ]

    if any(os.path.exists(path) for path in candidates):
        test_present.append(entry["run_id"])

if test_present:
    raise RuntimeError(
        "TEST DATA IS PRESENT IN RUNTIME:\n"
        + "\n".join(test_present)
    )

print("Physical test trajectories present: 0")
print("TEST SET IS SEALED.")
print()


# 
# Resolve sample records
# 

def get_sample_records(entries, trajectory_offset=0):
    records = []

    for local_traj_idx, entry in enumerate(entries):
        relative_sample_dir = entry["sample_dir"].replace("\\", "/")
        sample_dir = os.path.join(ROOT, relative_sample_dir)

        if not os.path.isdir(sample_dir):
            raise FileNotFoundError(
                f"{entry['run_id']}: sample directory not found:\n{sample_dir}"
            )

        files = sorted(
            os.path.join(sample_dir, filename)
            for filename in os.listdir(sample_dir)
            if filename.startswith("sample_") and filename.endswith(".npz")
        )

        expected = int(entry["num_samples"])

        if len(files) != expected:
            raise RuntimeError(
                f"{entry['run_id']}: expected {expected} samples, got {len(files)}"
            )

        traj_idx = trajectory_offset + local_traj_idx

        for sample_idx, path in enumerate(files):
            records.append(
                {
                    "path": path,
                    "traj_idx": traj_idx,
                    "run_id": entry["run_id"],
                    "sample_idx": sample_idx,
                }
            )

    return records


train_records = get_sample_records(train_runs, trajectory_offset=0)
val_records = get_sample_records(val_runs, trajectory_offset=0)

assert len(train_records) == 6000
assert len(val_records) == 6000

print("Train samples:", len(train_records))
print("Val samples:  ", len(val_records))
print("First train sample:", train_records[0]["path"])
print()


# 
# Cached graph
# 

class CachedGraph:
    __slots__ = (
        "atomic_numbers",
        "raw_types",
        "velocity",
        "force",
        "edge_index",
        "edge_vector",
        "edge_distance",
        "y",
        "traj_idx",
        "run_id",
        "sample_idx",
    )

    def __init__(
        self,
        atomic_numbers,
        raw_types,
        velocity,
        force,
        edge_index,
        edge_vector,
        edge_distance,
        y,
        traj_idx,
        run_id,
        sample_idx,
    ):
        self.atomic_numbers = atomic_numbers
        self.raw_types = raw_types
        self.velocity = velocity
        self.force = force
        self.edge_index = edge_index
        self.edge_vector = edge_vector
        self.edge_distance = edge_distance
        self.y = y
        self.traj_idx = traj_idx
        self.run_id = run_id
        self.sample_idx = sample_idx


# 
# Atomic species conversion
# 

def raw_types_to_atomic_numbers(raw_types):
    z = np.empty_like(raw_types, dtype=np.int64)

    for raw_type in np.unique(raw_types):
        raw_type_int = int(raw_type)

        if raw_type_int not in ATOMIC_NUMBER_BY_RAW_TYPE:
            raise RuntimeError(
                f"No atomic-number mapping for LAMMPS type {raw_type_int}."
            )

        z[raw_types == raw_type_int] = ATOMIC_NUMBER_BY_RAW_TYPE[raw_type_int]

    return z


# 
# Load cache + validate + training-only scale statistics
# 

def load_cache(records, collect_training_stats=False, label="cache"):
    cache = []

    vel_sq_total = 0.0
    force_sq_total = 0.0
    vector_component_count = 0

    vel_sum = np.zeros(3, dtype=np.float64)
    force_sum = np.zeros(3, dtype=np.float64)
    atom_count = 0

    seen_species = set()
    frame_strides = set()

    start = time.time()

    for i, record in enumerate(records):
        with np.load(record["path"]) as d:
            raw_types = d["atom_types"].astype(np.int64)
            atomic_numbers = raw_types_to_atomic_numbers(raw_types)

            velocity = np.stack(
                [d["vx_t"], d["vy_t"], d["vz_t"]],
                axis=1,
            ).astype(np.float32)

            force = np.stack(
                [d["fx_t"], d["fy_t"], d["fz_t"]],
                axis=1,
            ).astype(np.float32)

            src = d["edge_src"].astype(np.int64)
            dst = d["edge_dst"].astype(np.int64)

            edge_index = np.stack([src, dst], axis=0)

            edge_vector = d["edge_vector"].astype(np.float32)
            edge_distance = d["edge_distance"].astype(np.float32)

            y = d["non_affine_displacement"].astype(np.float32)

            if "timestep_t" in d and "timestep_t1" in d:
                t0 = int(np.asarray(d["timestep_t"]).item())
                t1 = int(np.asarray(d["timestep_t1"]).item())
                frame_strides.add(t1 - t0)

        n = raw_types.shape[0]

        if velocity.shape != (n, 3):
            raise RuntimeError(f"Bad velocity shape: {record['path']}")
        if force.shape != (n, 3):
            raise RuntimeError(f"Bad force shape: {record['path']}")
        if y.shape != (n, 3):
            raise RuntimeError(f"Bad target shape: {record['path']}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise RuntimeError(f"Bad edge_index shape: {record['path']}")
        if edge_vector.shape != (edge_index.shape[1], 3):
            raise RuntimeError(f"Bad edge_vector shape: {record['path']}")
        if edge_distance.shape[0] != edge_index.shape[1]:
            raise RuntimeError(f"Bad edge_distance shape: {record['path']}")

        arrays_to_check = [
            velocity,
            force,
            edge_vector,
            edge_distance,
            y,
        ]

        if not all(np.all(np.isfinite(x)) for x in arrays_to_check):
            raise RuntimeError(f"Non-finite values: {record['path']}")

        if np.any(edge_distance <= 0.0):
            raise RuntimeError(f"Non-positive edge distance: {record['path']}")

        if edge_index.size:
            if edge_index.min() < 0 or edge_index.max() >= n:
                raise RuntimeError(f"Out-of-range edge index: {record['path']}")

        # Geometry consistency check.
        reconstructed_distance = np.linalg.norm(edge_vector, axis=1)
        max_distance_error = float(
            np.max(np.abs(reconstructed_distance - edge_distance))
        )

        if max_distance_error > 2e-4:
            raise RuntimeError(
                f"edge_vector / edge_distance mismatch "
                f"{max_distance_error:.3e} Å: {record['path']}"
            )

        for z in np.unique(atomic_numbers):
            seen_species.add(int(z))

        if collect_training_stats:
            v64 = velocity.astype(np.float64)
            f64 = force.astype(np.float64)

            # IMPORTANT:
            # isotropic UN-CENTERED second moment:
            #
            # sigma_v^2 = (1 / (3N)) * sum_i ||v_i||^2
            #
            # No vector mean is subtracted anywhere.
            vel_sq_total += float(np.sum(v64 ** 2))
            force_sq_total += float(np.sum(f64 ** 2))
            vector_component_count += int(v64.size)

            # Means are diagnostics ONLY, never used in preprocessing.
            vel_sum += v64.sum(axis=0)
            force_sum += f64.sum(axis=0)
            atom_count += n

        cache.append(
            CachedGraph(
                atomic_numbers=torch.from_numpy(atomic_numbers),
                raw_types=torch.from_numpy(raw_types),
                velocity=torch.from_numpy(velocity),
                force=torch.from_numpy(force),
                edge_index=torch.from_numpy(edge_index),
                edge_vector=torch.from_numpy(edge_vector),
                edge_distance=torch.from_numpy(edge_distance),
                y=torch.from_numpy(y),
                traj_idx=int(record["traj_idx"]),
                run_id=record["run_id"],
                sample_idx=int(record["sample_idx"]),
            )
        )

        if (i + 1) % 1000 == 0:
            print(f"{label}: {i+1}/{len(records)}")

    elapsed = time.time() - start
    print(f"{label} loaded in {elapsed:.1f}s")

    stats = {
        "seen_species": sorted(seen_species),
        "frame_strides": sorted(frame_strides),
    }

    if collect_training_stats:
        velocity_scale = math.sqrt(
            vel_sq_total / vector_component_count
        )

        force_scale = math.sqrt(
            force_sq_total / vector_component_count
        )

        stats.update(
            {
                "velocity_scale": float(velocity_scale),
                "force_scale": float(force_scale),
                "velocity_mean_diagnostic": (
                    vel_sum / atom_count
                ).tolist(),
                "force_mean_diagnostic": (
                    force_sum / atom_count
                ).tolist(),
                "atom_count": int(atom_count),
                "vector_component_count": int(vector_component_count),
            }
        )

    return cache, stats


train_cache, train_stats = load_cache(
    train_records,
    collect_training_stats=True,
    label="Train cache",
)

val_cache, val_stats = load_cache(
    val_records,
    collect_training_stats=False,
    label="Val cache",
)

train_species = set(train_stats["seen_species"])
val_species = set(val_stats["seen_species"])

if not val_species.issubset(train_species):
    raise RuntimeError(
        f"Validation contains unseen species: {sorted(val_species - train_species)}"
    )

velocity_scale = float(train_stats["velocity_scale"])
force_scale = float(train_stats["force_scale"])

if velocity_scale <= 0.0 or force_scale <= 0.0:
    raise RuntimeError("Invalid vector normalization scale.")

all_frame_strides = set(train_stats["frame_strides"]) | set(
    val_stats["frame_strides"]
)

if all_frame_strides:
    if len(all_frame_strides) != 1:
        raise RuntimeError(
            f"Inconsistent transition strides: {sorted(all_frame_strides)}"
        )

    FRAME_STRIDE = next(iter(all_frame_strides))
else:
    FRAME_STRIDE = EXPECTED_FRAME_STRIDE
    print(
        "WARNING: timestep_t/timestep_t1 not present; "
        f"using configured stride {FRAME_STRIDE}."
    )

if FRAME_STRIDE != EXPECTED_FRAME_STRIDE:
    print(
        f"WARNING: observed frame stride {FRAME_STRIDE}, "
        f"configured expectation {EXPECTED_FRAME_STRIDE}."
    )

DELTA_T_PS = FRAME_STRIDE * MD_TIMESTEP_PS

print()
print("TRAIN-ONLY ISOTROPIC VECTOR SCALING")
print(
    "velocity sigma = sqrt(<||v||^2>/3):",
    f"{velocity_scale:.10g}",
)
print(
    "force sigma    = sqrt(<||f||^2>/3):",
    f"{force_scale:.10g}",
)
print(
    "Velocity mean (diagnostic only; NOT subtracted):",
    train_stats["velocity_mean_diagnostic"],
)
print(
    "Force mean (diagnostic only; NOT subtracted):",
    train_stats["force_mean_diagnostic"],
)
print("Active train species Z:", sorted(train_species))
print("Transition stride:", FRAME_STRIDE, "MD steps")
print("Transition dt:", DELTA_T_PS, "ps")
print("All 12,000 train+val graphs cached in RAM.")
print()


# 
# Dataset with isotropic no-centering preprocessing
# 

class AtomisticGraphDataset(torch.utils.data.Dataset):
    def __init__(self, cache, transform_matrix=None):
        self.cache = cache

        if transform_matrix is None:
            self.Q = None
        else:
            Q = np.asarray(transform_matrix, dtype=np.float32)

            if Q.shape != (3, 3):
                raise ValueError("Transform matrix must be 3x3.")

            orth_error = np.max(
                np.abs(Q @ Q.T - np.eye(3, dtype=np.float32))
            )

            if orth_error > 1e-5:
                raise ValueError(
                    f"Transform is not orthogonal; max error={orth_error:.3e}"
                )

            self.Q = torch.tensor(Q, dtype=torch.float32)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        g = self.cache[idx]

        velocity = g.velocity
        force = g.force
        edge_vector = g.edge_vector
        target = g.y

        if self.Q is not None:
            QT = self.Q.T

            # Transform physical vectors first.
            velocity = velocity @ QT
            force = force @ QT
            edge_vector = edge_vector @ QT
            target = target @ QT

        # One scalar scale per vector field.
        # No vector centering.
        velocity = velocity / velocity_scale
        force = force / force_scale

        return Data(
            atomic_numbers=g.atomic_numbers,
            raw_types=g.raw_types,
            velocity=velocity,
            force=force,
            edge_index=g.edge_index,
            edge_vector=edge_vector,
            edge_distance=g.edge_distance,
            y=target,
            traj_index=torch.tensor([g.traj_idx], dtype=torch.long),
            num_nodes=g.atomic_numbers.shape[0],
        )


train_dataset = AtomisticGraphDataset(train_cache)
val_dataset = AtomisticGraphDataset(val_cache)


# 
# DataLoader helper
# 

def make_loader(dataset, shuffle, seed, batch_size=BATCH_SIZE):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )


# 
# Metrics
# 

def basic_metric_dict(pred, target):
    pred = pred.float()
    target = target.float()

    error = pred - target

    mse = torch.mean(error ** 2).item()
    mae = torch.mean(torch.abs(error)).item()

    vector_error = torch.linalg.vector_norm(error, dim=-1)

    return {
        "component_mse": float(mse),
        "component_rmse": float(math.sqrt(mse)),
        "component_mae": float(mae),
        "vector_error_mean": float(vector_error.mean().item()),
        "vector_error_rms": float(
            torch.sqrt(torch.mean(vector_error ** 2)).item()
        ),
        "vector_error_median": float(
            torch.quantile(vector_error, 0.50).item()
        ),
        "vector_error_p95": float(
            torch.quantile(vector_error, 0.95).item()
        ),
        "vector_error_max": float(vector_error.max().item()),
    }


def detailed_metric_dict(
    pred,
    target,
    trajectory_index,
    atomic_numbers,
    trajectory_names,
):
    metrics = basic_metric_dict(pred, target)

    error = pred.float() - target.float()

    unique_traj = torch.unique(trajectory_index).tolist()
    per_trajectory = {}

    trajectory_rmses = []

    for traj_idx in unique_traj:
        mask = trajectory_index == traj_idx
        e = error[mask]

        rmse = float(
            torch.sqrt(torch.mean(e ** 2)).item()
        )

        vec = torch.linalg.vector_norm(e, dim=-1)

        name = trajectory_names[int(traj_idx)]

        per_trajectory[name] = {
            "component_rmse": rmse,
            "component_mae": float(torch.mean(torch.abs(e)).item()),
            "vector_error_mean": float(vec.mean().item()),
            "vector_error_p95": float(torch.quantile(vec, 0.95).item()),
            "n_atoms_over_transitions": int(mask.sum().item()),
        }

        trajectory_rmses.append(rmse)

    trajectory_rmses_np = np.asarray(trajectory_rmses, dtype=np.float64)

    metrics["trajectory_component_rmse_mean"] = float(
        trajectory_rmses_np.mean()
    )
    metrics["trajectory_component_rmse_std"] = float(
        trajectory_rmses_np.std(ddof=1)
        if trajectory_rmses_np.size > 1
        else 0.0
    )
    metrics["trajectory_count"] = int(trajectory_rmses_np.size)
    metrics["per_trajectory"] = per_trajectory

    per_species = {}

    for z in torch.unique(atomic_numbers).tolist():
        mask = atomic_numbers == z
        e = error[mask]
        vec = torch.linalg.vector_norm(e, dim=-1)

        label = ELEMENT_BY_Z.get(int(z), f"Z{int(z)}")

        per_species[label] = {
            "Z": int(z),
            "component_rmse": float(
                torch.sqrt(torch.mean(e ** 2)).item()
            ),
            "component_mae": float(torch.mean(torch.abs(e)).item()),
            "vector_error_mean": float(vec.mean().item()),
            "vector_error_p95": float(torch.quantile(vec, 0.95).item()),
            "n_atom_instances": int(mask.sum().item()),
        }

    metrics["per_species"] = per_species

    return metrics


# 
# Zero baseline
# 

def evaluate_zero_baseline(cache, trajectory_names):
    predictions = []
    targets = []
    traj = []
    species = []

    for g in cache:
        predictions.append(torch.zeros_like(g.y))
        targets.append(g.y)
        traj.append(
            torch.full(
                (g.y.shape[0],),
                g.traj_idx,
                dtype=torch.long,
            )
        )
        species.append(g.atomic_numbers)

    return detailed_metric_dict(
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(traj, dim=0),
        torch.cat(species, dim=0),
        trajectory_names,
    )


val_trajectory_names = {
    idx: entry["run_id"]
    for idx, entry in enumerate(val_runs)
}

zero_metrics = evaluate_zero_baseline(
    val_cache,
    val_trajectory_names,
)

print("VAL ZERO-PREDICTOR BASELINE")
print("Component RMSE:", f"{zero_metrics['component_rmse']:.8f} Å")
print("Component MAE: ", f"{zero_metrics['component_mae']:.8f} Å")
print()


# 
# Non-learned physics baselines
# 

EV_PER_ANGSTROM_TO_NEWTON = 1.602176634e-9
AMU_TO_KG = 1.66053906660e-27
ANGSTROM_TO_METER = 1e-10


def physics_baseline_predictions(cache, mode):
    predictions = []
    targets = []
    traj = []
    species = []

    dt_ps = DELTA_T_PS
    dt_s = dt_ps * 1e-12

    for g in cache:
        v = g.velocity.float()
        f = g.force.float()
        z = g.atomic_numbers.long()

        if mode == "zero":
            pred = torch.zeros_like(g.y)

        elif mode == "ballistic":
            # LAMMPS metal:
            # velocity = Angstrom / ps
            pred = v * dt_ps

        elif mode == "verlet_taylor":
            mass_amu = torch.tensor(
                [MASS_AMU_BY_Z[int(zi)] for zi in z.tolist()],
                dtype=torch.float32,
            )

            mass_kg = mass_amu * AMU_TO_KG

            acceleration_m_s2 = (
                f * EV_PER_ANGSTROM_TO_NEWTON
                / mass_kg[:, None]
            )

            acceleration_displacement_angstrom = (
                0.5
                * acceleration_m_s2
                * (dt_s ** 2)
                / ANGSTROM_TO_METER
            )

            pred = (
                v * dt_ps
                + acceleration_displacement_angstrom
            )

        else:
            raise ValueError(mode)

        predictions.append(pred)
        targets.append(g.y)
        traj.append(
            torch.full(
                (g.y.shape[0],),
                g.traj_idx,
                dtype=torch.long,
            )
        )
        species.append(z)

    return (
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(traj, dim=0),
        torch.cat(species, dim=0),
    )


physics_baseline_results = {}

if RUN_PHYSICS_BASELINES:
    print("=" * 80)
    print("NON-LEARNED PHYSICS BASELINES")
    print("=" * 80)

    for mode in ["zero", "ballistic", "verlet_taylor"]:
        pred, target, traj, z = physics_baseline_predictions(
            val_cache,
            mode,
        )

        metrics = detailed_metric_dict(
            pred,
            target,
            traj,
            z,
            val_trajectory_names,
        )

        physics_baseline_results[mode] = metrics

        print(
            f"{mode:14s} | "
            f"RMSE {metrics['component_rmse']:.8f} Å | "
            f"MAE {metrics['component_mae']:.8f} Å | "
            f"vector mean {metrics['vector_error_mean']:.8f} Å"
        )

    print()


# 
# Smooth radial basis
# 

def radial_basis(
    distance,
    cutoff=CUTOFF,
    num_basis=NUM_RADIAL,
):
    x = distance / cutoff

    centers = torch.linspace(
        0.0,
        1.0,
        num_basis,
        device=distance.device,
        dtype=distance.dtype,
    )

    width = 1.0 / num_basis

    basis = torch.exp(
        -0.5
        * (
            (x[:, None] - centers[None, :])
            / width
        ) ** 2
    )

    envelope = 0.5 * (
        torch.cos(
            math.pi
            * torch.clamp(x, 0.0, 1.0)
        )
        + 1.0
    )

    envelope = envelope * (
        x <= 1.0
    ).to(distance.dtype)

    return basis * envelope[:, None]


# 
# E3 convolution
# 

class EquivariantConv(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = EDGE_IRREPS

        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_edge,
            self.irreps_out,
            shared_weights=False,
            internal_weights=False,
        )

        self.radial_mlp = nn.Sequential(
            nn.Linear(NUM_RADIAL, RADIAL_HIDDEN),
            nn.SiLU(),
            nn.Linear(
                RADIAL_HIDDEN,
                self.tp.weight_numel,
            ),
        )

    def forward(
        self,
        x,
        edge_index,
        edge_vector,
        edge_distance,
    ):
        src = edge_index[0]
        dst = edge_index[1]

        edge_sh = o3.spherical_harmonics(
            self.irreps_edge,
            edge_vector,
            normalize=True,
            normalization="component",
        )

        radial = radial_basis(edge_distance)
        weights = self.radial_mlp(radial)

        messages = self.tp(
            x[src],
            edge_sh,
            weights,
        )

        out = torch.zeros(
            (
                x.shape[0],
                self.irreps_out.dim,
            ),
            dtype=x.dtype,
            device=x.device,
        )

        out.index_add_(0, dst, messages)

        degree = torch.zeros(
            x.shape[0],
            dtype=x.dtype,
            device=x.device,
        )

        degree.index_add_(
            0,
            dst,
            torch.ones(
                dst.shape[0],
                dtype=x.dtype,
                device=x.device,
            ),
        )

        # Mean aggregation to match NNConv's aggregation rule.
        out = out / degree.clamp(min=1.0)[:, None]

        return out


# 
# Equivariant activation
# 

class EquivariantActivation(nn.Module):
    """
    Scalars: SiLU.

    Non-scalars:
        x -> sigmoid(a ||x|| + b) * x

    ||x|| is O(3)-invariant, so scalar gating preserves the
    input irrep and parity.
    """

    def __init__(self, irreps):
        super().__init__()

        self.irreps = o3.Irreps(irreps)

        self.tensor_gates = nn.ParameterList()
        self.tensor_biases = nn.ParameterList()

        for mul, ir in self.irreps:
            if ir.l > 0:
                self.tensor_gates.append(
                    nn.Parameter(torch.ones(mul))
                )
                self.tensor_biases.append(
                    nn.Parameter(torch.zeros(mul))
                )

    def forward(self, x):
        pieces = []

        offset = 0
        tensor_index = 0

        for mul, ir in self.irreps:
            block_dim = mul * ir.dim

            block = x[
                :,
                offset:offset + block_dim,
            ].reshape(
                x.shape[0],
                mul,
                ir.dim,
            )

            if ir.l == 0:
                block = F.silu(block)

            else:
                norms = torch.sqrt(
                    torch.sum(block ** 2, dim=-1)
                    + 1e-12
                )

                gate = torch.sigmoid(
                    norms
                    * self.tensor_gates[
                        tensor_index
                    ][None, :]
                    + self.tensor_biases[
                        tensor_index
                    ][None, :]
                )

                block = block * gate[:, :, None]
                tensor_index += 1

            pieces.append(
                block.reshape(
                    x.shape[0],
                    block_dim,
                )
            )

            offset += block_dim

        return torch.cat(pieces, dim=-1)


# 
# E3 A/B/C model
# 

def input_irreps_for_variant(variant):
    if variant == "A":
        return o3.Irreps(
            f"{SPECIES_EMBED_DIM}x0e"
        )

    if variant == "B":
        return o3.Irreps(
            f"{SPECIES_EMBED_DIM}x0e + 1x1o"
        )

    if variant == "C":
        return o3.Irreps(
            f"{SPECIES_EMBED_DIM}x0e + 2x1o"
        )

    raise ValueError(variant)


class E3DynamicsGNN(nn.Module):
    def __init__(self, variant):
        super().__init__()

        if variant not in {"A", "B", "C"}:
            raise ValueError(variant)

        self.variant = variant

        self.species_embedding = nn.Embedding(
            MAX_ATOMIC_NUMBER + 1,
            SPECIES_EMBED_DIM,
        )

        self.irreps_input = input_irreps_for_variant(
            variant
        )
        self.irreps_hidden = HIDDEN_IRREPS
        self.irreps_output = OUTPUT_IRREPS

        self.input_linear = o3.Linear(
            self.irreps_input,
            self.irreps_hidden,
        )

        self.conv1 = EquivariantConv(
            self.irreps_hidden,
            self.irreps_hidden,
        )
        self.act1 = EquivariantActivation(
            self.irreps_hidden
        )

        self.conv2 = EquivariantConv(
            self.irreps_hidden,
            self.irreps_hidden,
        )
        self.act2 = EquivariantActivation(
            self.irreps_hidden
        )

        self.output_linear = o3.Linear(
            self.irreps_hidden,
            self.irreps_output,
        )

    def build_input(self, data):
        species = self.species_embedding(
            data.atomic_numbers
        )

        if self.variant == "A":
            return species

        if self.variant == "B":
            return torch.cat(
                [species, data.velocity],
                dim=-1,
            )

        return torch.cat(
            [
                species,
                data.velocity,
                data.force,
            ],
            dim=-1,
        )

    def forward(self, data):
        x = self.build_input(data)
        x = self.input_linear(x)

        # Two stable equivariant residual message-passing blocks.
        message = self.conv1(
            x,
            data.edge_index,
            data.edge_vector,
            data.edge_distance,
        )
        x = self.act1(x + message)

        message = self.conv2(
            x,
            data.edge_index,
            data.edge_vector,
            data.edge_distance,
        )
        x = self.act2(x + message)

        return self.output_linear(x)


# 
# Node-only O(3)-equivariant baseline
# 

class NodeOnlyEquivariantBaseline(nn.Module):
    """
    No graph / no edges.

    B:
        alpha(species, |v|) * v

    C:
        alpha(species, |v|, |f|, v.f) * v
      + beta(species, |v|, |f|, v.f) * f

    Coefficients are scalar functions of O(3)-invariants, so
    the output is exactly O(3)-equivariant.
    """

    def __init__(self, variant):
        super().__init__()

        if variant not in {"B", "C"}:
            raise ValueError(
                "Node-only equivariant baseline supports B or C."
            )

        self.variant = variant

        self.species_embedding = nn.Embedding(
            MAX_ATOMIC_NUMBER + 1,
            SPECIES_EMBED_DIM,
        )

        invariant_dim = (
            SPECIES_EMBED_DIM + 1
            if variant == "B"
            else SPECIES_EMBED_DIM + 3
        )

        output_coefficients = 1 if variant == "B" else 2

        self.mlp = nn.Sequential(
            nn.Linear(
                invariant_dim,
                NODE_BASELINE_HIDDEN,
            ),
            nn.SiLU(),
            nn.Linear(
                NODE_BASELINE_HIDDEN,
                NODE_BASELINE_HIDDEN,
            ),
            nn.SiLU(),
            nn.Linear(
                NODE_BASELINE_HIDDEN,
                output_coefficients,
            ),
        )

    def forward(self, data):
        species = self.species_embedding(
            data.atomic_numbers
        )

        v = data.velocity
        v_norm = torch.linalg.vector_norm(
            v,
            dim=-1,
            keepdim=True,
        )

        if self.variant == "B":
            invariants = torch.cat(
                [species, v_norm],
                dim=-1,
            )

            alpha = self.mlp(invariants)
            return alpha * v

        f = data.force

        f_norm = torch.linalg.vector_norm(
            f,
            dim=-1,
            keepdim=True,
        )

        v_dot_f = torch.sum(
            v * f,
            dim=-1,
            keepdim=True,
        )

        invariants = torch.cat(
            [
                species,
                v_norm,
                f_norm,
                v_dot_f,
            ],
            dim=-1,
        )

        coefficients = self.mlp(invariants)

        alpha = coefficients[:, 0:1]
        beta = coefficients[:, 1:2]

        return alpha * v + beta * f


# 
# Matched-preprocessing NNConv-C
# 

class MatchedNNConvC(nn.Module):
    """
    Conventional non-equivariant GNN.

    IMPORTANT:
    This baseline uses the SAME:
      - atomic-number categorical embedding
      - isotropic velocity scaling
      - isotropic force scaling
      - no vector mean subtraction
      - train/val split
      - mean aggregation
      - two message passing layers

    Therefore its symmetry stress test isolates architecture
    much more cleanly than the old component-normalized NNConv.
    """

    def __init__(self):
        super().__init__()

        self.species_embedding = nn.Embedding(
            MAX_ATOMIC_NUMBER + 1,
            SPECIES_EMBED_DIM,
        )

        input_dim = SPECIES_EMBED_DIM + 3 + 3

        self.input = nn.Linear(
            input_dim,
            NN_HIDDEN,
        )

        edge_net1 = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(
                64,
                NN_HIDDEN * NN_HIDDEN,
            ),
        )

        self.conv1 = NNConv(
            NN_HIDDEN,
            NN_HIDDEN,
            edge_net1,
            aggr="mean",
        )

        edge_net2 = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(
                64,
                NN_HIDDEN * NN_HIDDEN,
            ),
        )

        self.conv2 = NNConv(
            NN_HIDDEN,
            NN_HIDDEN,
            edge_net2,
            aggr="mean",
        )

        self.head = nn.Sequential(
            nn.Linear(
                NN_HIDDEN,
                NN_HIDDEN,
            ),
            nn.SiLU(),
            nn.Linear(
                NN_HIDDEN,
                3,
            ),
        )

    def forward(self, data):
        species = self.species_embedding(
            data.atomic_numbers
        )

        x = torch.cat(
            [
                species,
                data.velocity,
                data.force,
            ],
            dim=-1,
        )

        x = F.silu(self.input(x))

        direction = (
            data.edge_vector
            / data.edge_distance.clamp(
                min=1e-8
            )[:, None]
        )

        edge_attr = torch.cat(
            [
                direction,
                (
                    data.edge_distance[:, None]
                    / CUTOFF
                ),
            ],
            dim=-1,
        )

        residual = x

        x = self.conv1(
            x,
            data.edge_index,
            edge_attr,
        )
        x = F.silu(x)

        x = self.conv2(
            x,
            data.edge_index,
            edge_attr,
        )

        x = F.silu(x + residual)

        return self.head(x)


# 
# Transform generation
# 

def random_so3_matrix(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)

    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0

    Q = Q @ np.diag(signs)

    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0

    return Q.astype(np.float32)


rng = np.random.default_rng(ROTATION_SEED)

SO3_TRANSFORMS = []

for i in range(N_SO3_ROTATIONS):
    Q = random_so3_matrix(rng)

    SO3_TRANSFORMS.append(
        {
            "name": f"SO3_{i+1:02d}",
            "kind": "proper",
            "matrix": Q,
        }
    )


REFLECT_X = np.diag(
    [-1.0, 1.0, 1.0]
).astype(np.float32)

INVERSION = (
    -np.eye(3, dtype=np.float32)
)

O3_IMPROPER_TRANSFORMS = [
    {
        "name": "reflect_x",
        "kind": "improper",
        "matrix": REFLECT_X,
    },
    {
        "name": "inversion",
        "kind": "improper",
        "matrix": INVERSION,
    },
]

for i in range(N_RANDOM_IMPROPER):
    R = random_so3_matrix(rng)
    Q = R @ REFLECT_X

    O3_IMPROPER_TRANSFORMS.append(
        {
            "name": f"improper_random_{i+1:02d}",
            "kind": "improper",
            "matrix": Q.astype(np.float32),
        }
    )

ALL_STRESS_TRANSFORMS = (
    SO3_TRANSFORMS
    + O3_IMPROPER_TRANSFORMS
)

for item in ALL_STRESS_TRANSFORMS:
    determinant = float(
        np.linalg.det(item["matrix"])
    )

    if item["kind"] == "proper":
        assert determinant > 0.999
    else:
        assert determinant < -0.999


# 
# Prediction helper
# 

@torch.no_grad()
def predict_dataset(
    model,
    dataset,
    use_amp=False,
    batch_size=BATCH_SIZE,
):
    loader = make_loader(
        dataset,
        shuffle=False,
        seed=0,
        batch_size=batch_size,
    )

    model.eval()

    predictions = []
    targets = []
    trajectory_index = []
    atomic_numbers = []

    for batch in loader:
        batch = batch.to(
            DEVICE,
            non_blocking=True,
        )

        if use_amp and DEVICE.type == "cuda":
            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
            ):
                pred = model(batch)
        else:
            pred = model(batch)

        predictions.append(
            pred.float().cpu()
        )

        targets.append(
            batch.y.float().cpu()
        )

        # batch.traj_index is graph-level.
        # batch.batch maps each node to its graph in the batch.
        node_traj = (
            batch.traj_index[
                batch.batch
            ]
            .long()
            .cpu()
        )

        trajectory_index.append(node_traj)

        atomic_numbers.append(
            batch.atomic_numbers
            .long()
            .cpu()
        )

    return (
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(trajectory_index, dim=0),
        torch.cat(atomic_numbers, dim=0),
    )


# 
# E3 numerical equivariance self-test
# 

@torch.no_grad()
def e3_equivariance_self_test(model, variant):
    model.eval()

    tiny_cache = val_cache[:SELFTEST_GRAPHS]

    base_dataset = AtomisticGraphDataset(
        tiny_cache
    )

    base_pred, _, _, _ = predict_dataset(
        model,
        base_dataset,
        use_amp=False,
        batch_size=SELFTEST_GRAPHS,
    )

    selftest_transforms = [
        {
            "name": "proper_rotation",
            "matrix": SO3_TRANSFORMS[0]["matrix"],
        },
        {
            "name": "improper_reflection",
            "matrix": REFLECT_X,
        },
    ]

    results = {}

    for item in selftest_transforms:
        Q = item["matrix"]

        transformed_dataset = (
            AtomisticGraphDataset(
                tiny_cache,
                transform_matrix=Q,
            )
        )

        transformed_pred, _, _, _ = predict_dataset(
            model,
            transformed_dataset,
            use_amp=False,
            batch_size=SELFTEST_GRAPHS,
        )

        Q_tensor = torch.tensor(
            Q,
            dtype=torch.float32,
        )

        expected = base_pred @ Q_tensor.T
        diff = transformed_pred - expected

        rmse = float(
            torch.sqrt(
                torch.mean(diff ** 2)
            ).item()
        )

        max_abs = float(
            torch.max(torch.abs(diff)).item()
        )

        results[item["name"]] = {
            "rmse": rmse,
            "max_abs": max_abs,
            "determinant": float(
                np.linalg.det(Q)
            ),
        }

        print(
            f"E3 preflight {variant} "
            f"{item['name']:20s} | "
            f"RMSE {rmse:.3e} | "
            f"max {max_abs:.3e}"
        )

        if (
            rmse > EQ_SELFTEST_RMSE_TOL
            or max_abs > EQ_SELFTEST_MAX_TOL
        ):
            raise RuntimeError(
                f"E3 equivariance preflight FAILED "
                f"for variant {variant}, {item['name']}."
            )

    return results


# 
# Basic evaluation during training
# 

@torch.no_grad()
def evaluate_basic(
    model,
    loader,
    use_amp=False,
):
    model.eval()

    sum_squared = 0.0
    sum_absolute = 0.0
    n_values = 0

    for batch in loader:
        batch = batch.to(
            DEVICE,
            non_blocking=True,
        )

        if use_amp and DEVICE.type == "cuda":
            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
            ):
                pred = model(batch)
        else:
            pred = model(batch)

        error = (
            pred.float()
            - batch.y.float()
        )

        sum_squared += torch.sum(
            error ** 2
        ).item()

        sum_absolute += torch.sum(
            torch.abs(error)
        ).item()

        n_values += error.numel()

    mse = sum_squared / n_values

    return (
        float(mse),
        float(math.sqrt(mse)),
        float(sum_absolute / n_values),
    )


# 
# Generic training
# 

def train_model(
    model_factory,
    experiment_name,
    seed,
    use_amp=False,
    e3_variant=None,
):
    print()
    print("=" * 80)
    print(experiment_name)
    print("=" * 80)

    set_seed(seed)

    train_loader = make_loader(
        train_dataset,
        shuffle=True,
        seed=seed,
    )

    val_loader = make_loader(
        val_dataset,
        shuffle=False,
        seed=seed,
    )

    model = model_factory().to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Parameters:", parameter_count)
    print("Seed:", seed)

    preflight = None

    if e3_variant is not None:
        preflight = e3_equivariance_self_test(
            model,
            e3_variant,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    amp_enabled = (
        bool(use_amp)
        and DEVICE.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    best_val_mse = float("inf")
    best_epoch = 0
    evals_without_improvement = 0

    safe_name = (
        experiment_name
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    checkpoint = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_seed{seed}.pt",
    )

    history = []
    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        train_squared = 0.0
        train_values = 0

        for batch in train_loader:
            batch = batch.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            if amp_enabled:
                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.float16,
                ):
                    pred = model(batch)
                    loss = criterion(
                        pred,
                        batch.y,
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss in {experiment_name}, "
                        f"epoch {epoch}."
                    )

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP_NORM,
                )

                scaler.step(optimizer)
                scaler.update()

            else:
                pred = model(batch)

                loss = criterion(
                    pred,
                    batch.y,
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss in {experiment_name}, "
                        f"epoch {epoch}."
                    )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP_NORM,
                )

                optimizer.step()

            error = (
                pred.detach().float()
                - batch.y.float()
            )

            train_squared += torch.sum(
                error ** 2
            ).item()

            train_values += error.numel()

        train_mse = (
            train_squared
            / train_values
        )
        train_rmse = math.sqrt(train_mse)

        do_validation = (
            epoch == 1
            or epoch % VAL_EVERY == 0
            or epoch == MAX_EPOCHS
        )

        val_mse = None
        val_rmse = None
        val_mae = None

        if do_validation:
            (
                val_mse,
                val_rmse,
                val_mae,
            ) = evaluate_basic(
                model,
                val_loader,
                use_amp=False,
            )

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                evals_without_improvement = 0

                torch.save(
                    {
                        "model_state_dict":
                            model.state_dict(),
                        "experiment_name":
                            experiment_name,
                        "seed":
                            seed,
                        "epoch":
                            epoch,
                        "val_mse":
                            val_mse,
                        "parameter_count":
                            parameter_count,
                    },
                    checkpoint,
                )

            else:
                evals_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_rmse":
                    float(train_rmse),
                "val_rmse":
                    None
                    if val_rmse is None
                    else float(val_rmse),
                "val_mae":
                    None
                    if val_mae is None
                    else float(val_mae),
            }
        )

        if do_validation:
            print(
                f"Epoch {epoch:03d} | "
                f"train RMSE {train_rmse:.6e} Å | "
                f"val RMSE {val_rmse:.6e} Å | "
                f"val MAE {val_mae:.6e} Å | "
                f"{time.time()-epoch_start:.1f}s"
            )

        if (
            do_validation
            and evals_without_improvement
            >= PATIENCE_EVALS
        ):
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch {best_epoch}."
            )
            break

    elapsed = time.time() - start

    checkpoint_payload = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint_payload[
            "model_state_dict"
        ]
    )

    (
        pred,
        target,
        traj_idx,
        atomic_numbers,
    ) = predict_dataset(
        model,
        val_dataset,
        use_amp=False,
    )

    final_metrics = detailed_metric_dict(
        pred,
        target,
        traj_idx,
        atomic_numbers,
        val_trajectory_names,
    )

    checkpoint_rmse_difference = abs(
        final_metrics[
            "component_rmse"
        ]
        - math.sqrt(best_val_mse)
    )

    if checkpoint_rmse_difference > 5e-6:
        raise RuntimeError(
            f"Checkpoint reproduction failed for {experiment_name}: "
            f"difference={checkpoint_rmse_difference:.3e} Å"
        )

    epoch_cap_warning = (
        best_epoch
        >= MAX_EPOCHS - VAL_EVERY
    )

    if epoch_cap_warning:
        print(
            "WARNING: best checkpoint occurred near the epoch cap; "
            "this model may still be improving."
        )

    history_file = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_seed{seed}_history.json",
    )

    with open(history_file, "w") as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    result = {
        "experiment_name":
            experiment_name,
        "seed":
            int(seed),
        "parameters":
            int(parameter_count),
        "best_epoch":
            int(best_epoch),
        "wall_seconds":
            float(elapsed),
        "checkpoint":
            checkpoint,
        "checkpoint_rmse_difference":
            float(checkpoint_rmse_difference),
        "epoch_cap_warning":
            bool(epoch_cap_warning),
        "preflight_equivariance":
            preflight,
        "validation":
            final_metrics,
    }

    print()
    print("BEST:", experiment_name)
    print("Best epoch:", best_epoch)
    print(
        "Validation component RMSE:",
        f"{final_metrics['component_rmse']:.8f} Å",
    )
    print(
        "Validation component MAE: ",
        f"{final_metrics['component_mae']:.8f} Å",
    )
    print(
        "Validation vector mean error:",
        f"{final_metrics['vector_error_mean']:.8f} Å",
    )
    print(
        "Trajectory RMSE mean ± SD:",
        f"{final_metrics['trajectory_component_rmse_mean']:.8f} "
        f"± {final_metrics['trajectory_component_rmse_std']:.8f} Å",
    )
    print(
        "Wall time:",
        f"{elapsed/60.0:.2f} min",
    )

    del model
    del optimizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# 
# Load a trained result
# 

def load_model_from_result(
    result,
    model_factory,
):
    model = model_factory().to(DEVICE)

    payload = torch.load(
        result["checkpoint"],
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        payload["model_state_dict"]
    )

    model.eval()
    return model


# 
# Symmetry stress test
# 

def symmetry_stress_test(
    result,
    model_factory,
    use_amp=False,
):
    model = load_model_from_result(
        result,
        model_factory,
    )

    (
        original_pred,
        original_target,
        _,
        _,
    ) = predict_dataset(
        model,
        val_dataset,
        use_amp=False,
    )

    original_metrics = basic_metric_dict(
        original_pred,
        original_target,
    )

    reproduction_difference = abs(
        original_metrics[
            "component_rmse"
        ]
        - result["validation"][
            "component_rmse"
        ]
    )

    if reproduction_difference > 5e-6:
        raise RuntimeError(
            f"Stress-test checkpoint reproduction failed: "
            f"{result['experiment_name']}"
        )

    transform_results = []

    print()
    print("=" * 80)
    print(
        "SYMMETRY STRESS TEST:",
        result["experiment_name"],
    )
    print("=" * 80)
    print(
        "Original RMSE:",
        f"{original_metrics['component_rmse']:.8f} Å",
    )

    for item in ALL_STRESS_TRANSFORMS:
        Q = item["matrix"]

        transformed_dataset = (
            AtomisticGraphDataset(
                val_cache,
                transform_matrix=Q,
            )
        )

        (
            transformed_pred,
            transformed_target,
            _,
            _,
        ) = predict_dataset(
            model,
            transformed_dataset,
            use_amp=False,
        )

        accuracy = basic_metric_dict(
            transformed_pred,
            transformed_target,
        )

        Q_tensor = torch.tensor(
            Q,
            dtype=torch.float32,
        )

        expected_transformed_pred = (
            original_pred
            @ Q_tensor.T
        )

        equivariance = basic_metric_dict(
            transformed_pred,
            expected_transformed_pred,
        )

        degradation = (
            accuracy["component_rmse"]
            / original_metrics[
                "component_rmse"
            ]
        )

        row = {
            "name":
                item["name"],
            "kind":
                item["kind"],
            "matrix":
                Q.tolist(),
            "determinant":
                float(np.linalg.det(Q)),
            "target_component_rmse":
                accuracy["component_rmse"],
            "target_component_mae":
                accuracy["component_mae"],
            "equivariance_component_rmse":
                equivariance[
                    "component_rmse"
                ],
            "equivariance_component_mae":
                equivariance[
                    "component_mae"
                ],
            "accuracy_degradation_factor":
                float(degradation),
        }

        transform_results.append(row)

        print(
            f"{item['name']:20s} | "
            f"det {row['determinant']:+.1f} | "
            f"target RMSE "
            f"{row['target_component_rmse']:.8f} Å | "
            f"equiv RMSE "
            f"{row['equivariance_component_rmse']:.3e} Å | "
            f"degradation {degradation:.5f}x"
        )

    def summarize(kind):
        rows = [
            row
            for row in transform_results
            if row["kind"] == kind
        ]

        target_rmse = np.asarray(
            [
                row["target_component_rmse"]
                for row in rows
            ],
            dtype=np.float64,
        )

        eq_rmse = np.asarray(
            [
                row["equivariance_component_rmse"]
                for row in rows
            ],
            dtype=np.float64,
        )

        degradation = np.asarray(
            [
                row["accuracy_degradation_factor"]
                for row in rows
            ],
            dtype=np.float64,
        )

        return {
            "n_transforms":
                int(len(rows)),
            "target_rmse_mean":
                float(target_rmse.mean()),
            "target_rmse_std":
                float(
                    target_rmse.std(ddof=1)
                    if len(rows) > 1
                    else 0.0
                ),
            "equivariance_rmse_mean":
                float(eq_rmse.mean()),
            "equivariance_rmse_std":
                float(
                    eq_rmse.std(ddof=1)
                    if len(rows) > 1
                    else 0.0
                ),
            "degradation_mean":
                float(degradation.mean()),
            "degradation_min":
                float(degradation.min()),
            "degradation_max":
                float(degradation.max()),
        }

    summary = {
        "experiment_name":
            result["experiment_name"],
        "seed":
            result["seed"],
        "original_component_rmse":
            original_metrics[
                "component_rmse"
            ],
        "checkpoint_reproduction_difference":
            float(reproduction_difference),
        "proper_SO3":
            summarize("proper"),
        "improper_O3":
            summarize("improper"),
        "transforms":
            transform_results,
    }

    print()
    print(
        "SO(3) equivariance RMSE mean:",
        f"{summary['proper_SO3']['equivariance_rmse_mean']:.3e} Å",
    )
    print(
        "SO(3) degradation mean:",
        f"{summary['proper_SO3']['degradation_mean']:.6f}x",
    )
    print(
        "Improper O(3) equivariance RMSE mean:",
        f"{summary['improper_O3']['equivariance_rmse_mean']:.3e} Å",
    )
    print(
        "Improper O(3) degradation mean:",
        f"{summary['improper_O3']['degradation_mean']:.6f}x",
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# 
# Train experiments
# 

experiment_start = time.time()

all_results = []
symmetry_results = []


# ------------------------------------------------------------
# Node-only O(3) baselines
# ------------------------------------------------------------

if RUN_NODE_EQUIV_BASELINES:
    for seed in TRAIN_SEEDS:
        for variant in ["B", "C"]:
            result = train_model(
                model_factory=(
                    lambda v=variant:
                    NodeOnlyEquivariantBaseline(v)
                ),
                experiment_name=(
                    f"node_only_equivariant_{variant}"
                ),
                seed=seed,
                use_amp=USE_AMP_FOR_NON_E3,
                e3_variant=None,
            )

            result["family"] = "node_only_equivariant"
            result["variant"] = variant

            all_results.append(result)

            if RUN_SYMMETRY_STRESS_TESTS:
                stress = symmetry_stress_test(
                    result,
                    model_factory=(
                        lambda v=variant:
                        NodeOnlyEquivariantBaseline(v)
                    ),
                    use_amp=USE_AMP_FOR_NON_E3,
                )

                symmetry_results.append(stress)


# ------------------------------------------------------------
# E3 A/B/C
# ------------------------------------------------------------

if RUN_E3_ABC:
    for seed in TRAIN_SEEDS:
        for variant in ["A", "B", "C"]:
            result = train_model(
                model_factory=(
                    lambda v=variant:
                    E3DynamicsGNN(v)
                ),
                experiment_name=(
                    f"e3_variant_{variant}"
                ),
                seed=seed,
                use_amp=False,
                e3_variant=variant,
            )

            result["family"] = "E3"
            result["variant"] = variant

            all_results.append(result)

            if RUN_SYMMETRY_STRESS_TESTS:
                stress = symmetry_stress_test(
                    result,
                    model_factory=(
                        lambda v=variant:
                        E3DynamicsGNN(v)
                    ),
                    use_amp=False,
                )

                symmetry_results.append(stress)


# ------------------------------------------------------------
# Matched NNConv-C
# ------------------------------------------------------------

if RUN_MATCHED_NNCONV_C:
    for seed in TRAIN_SEEDS:
        result = train_model(
            model_factory=MatchedNNConvC,
            experiment_name="matched_nnconv_C",
            seed=seed,
            use_amp=USE_AMP_FOR_NON_E3,
            e3_variant=None,
        )

        result["family"] = "matched_NNConv"
        result["variant"] = "C"

        all_results.append(result)

        if RUN_SYMMETRY_STRESS_TESTS:
            stress = symmetry_stress_test(
                result,
                model_factory=MatchedNNConvC,
                use_amp=USE_AMP_FOR_NON_E3,
            )

            symmetry_results.append(stress)


# 
# Summary helpers
# 

def find_stress(experiment_name, seed):
    for item in symmetry_results:
        if (
            item["experiment_name"]
            == experiment_name
            and item["seed"] == seed
        ):
            return item

    return None


# 
# Final summary
# 

print()
print("=" * 100)
print("FINAL FIXED EXPERIMENT SUMMARY")
print("=" * 100)

print()
print("PHYSICS BASELINES")

for name, metrics in physics_baseline_results.items():
    print(
        f"{name:14s} | "
        f"RMSE {metrics['component_rmse']:.8f} Å | "
        f"vector mean {metrics['vector_error_mean']:.8f} Å"
    )

print()
print("LEARNED MODELS")

for result in all_results:
    metrics = result["validation"]

    improvement_vs_zero = (
        1.0
        - metrics["component_rmse"]
        / zero_metrics["component_rmse"]
    ) * 100.0

    stress = find_stress(
        result["experiment_name"],
        result["seed"],
    )

    stress_text = ""

    if stress is not None:
        stress_text = (
            f" | SO3 eq "
            f"{stress['proper_SO3']['equivariance_rmse_mean']:.3e} Å"
            f" | SO3 deg "
            f"{stress['proper_SO3']['degradation_mean']:.5f}x"
            f" | O3-improper eq "
            f"{stress['improper_O3']['equivariance_rmse_mean']:.3e} Å"
        )

    print(
        f"{result['experiment_name']:28s} "
        f"seed {result['seed']:5d} | "
        f"params {result['parameters']:7d} | "
        f"epoch {result['best_epoch']:3d} | "
        f"RMSE {metrics['component_rmse']:.8f} Å | "
        f"vs-zero {improvement_vs_zero:+.2f}%"
        f"{stress_text}"
    )


# 
# Save full provenance
# 

total_minutes = (
    time.time()
    - experiment_start
) / 60.0

provenance = {
    "experiment":
        "anorthite_full_fixed_e3nn_abc",
    "timestamp_unix":
        time.time(),
    "device":
        str(DEVICE),
    "gpu":
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    "python_version":
        platform.python_version(),
    "torch_version":
        torch.__version__,
    "e3nn_version":
        e3nn.__version__,
    "torch_geometric_version":
        importlib.import_module(
            "torch_geometric"
        ).__version__,
    "train_seeds":
        TRAIN_SEEDS,
    "rotation_seed":
        ROTATION_SEED,
    "train_trajectories":
        train_run_ids,
    "val_trajectories":
        val_run_ids,
    "test_trajectories_manifest_only":
        test_run_ids,
    "test_set_used":
        False,
    "train_samples":
        len(train_records),
    "val_samples":
        len(val_records),
    "normalization": {
        "method":
            "uncentered_isotropic_rms",
        "velocity_formula":
            "sqrt(mean over train atoms/components of v^2)",
        "force_formula":
            "sqrt(mean over train atoms/components of f^2)",
        "velocity_scale":
            velocity_scale,
        "force_scale":
            force_scale,
        "vector_mean_subtracted":
            False,
        "velocity_mean_diagnostic":
            train_stats[
                "velocity_mean_diagnostic"
            ],
        "force_mean_diagnostic":
            train_stats[
                "force_mean_diagnostic"
            ],
    },
    "species": {
        "representation":
            "learned categorical embedding of atomic number",
        "embedding_dim":
            SPECIES_EMBED_DIM,
        "raw_type_to_atomic_number":
            ATOMIC_NUMBER_BY_RAW_TYPE,
        "active_train_species_Z":
            sorted(train_species),
    },
    "simulation_transition": {
        "lammps_units":
            "metal",
        "md_timestep_ps":
            MD_TIMESTEP_PS,
        "frame_stride_steps":
            FRAME_STRIDE,
        "delta_t_ps":
            DELTA_T_PS,
    },
    "graph": {
        "cutoff_angstrom":
            CUTOFF,
        "aggregation":
            "mean",
    },
    "optimization": {
        "batch_size":
            BATCH_SIZE,
        "max_epochs":
            MAX_EPOCHS,
        "learning_rate":
            LEARNING_RATE,
        "weight_decay":
            WEIGHT_DECAY,
        "validation_every":
            VAL_EVERY,
        "patience_evals":
            PATIENCE_EVALS,
        "grad_clip_norm":
            GRAD_CLIP_NORM,
        "E3_AMP":
            False,
        "non_E3_training_AMP":
            USE_AMP_FOR_NON_E3,
        "reported_validation_and_stress_precision":
            "FP32",
    },
    "E3_architecture": {
        "hidden_irreps":
            str(HIDDEN_IRREPS),
        "edge_irreps":
            str(EDGE_IRREPS),
        "output_irreps":
            str(OUTPUT_IRREPS),
        "num_radial":
            NUM_RADIAL,
        "radial_hidden":
            RADIAL_HIDDEN,
        "residual_blocks":
            2,
    },
    "symmetry_tests": {
        "n_SO3_rotations":
            len(SO3_TRANSFORMS),
        "n_improper_O3":
            len(
                O3_IMPROPER_TRANSFORMS
            ),
        "preflight_rmse_tolerance":
            EQ_SELFTEST_RMSE_TOL,
        "preflight_max_tolerance":
            EQ_SELFTEST_MAX_TOL,
    },
    "zero_baseline":
        zero_metrics,
    "physics_baselines":
        physics_baseline_results,
    "learned_results":
        all_results,
    "symmetry_results":
        symmetry_results,
    "total_minutes":
        float(total_minutes),
}

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "full_fixed_results.json",
)

with open(RESULT_FILE, "w") as f:
    json.dump(
        provenance,
        f,
        indent=2,
    )


# 
# Machine-readable compact CSV summary
# 

csv_file = os.path.join(
    OUTPUT_DIR,
    "model_summary.csv",
)

with open(csv_file, "w") as f:
    f.write(
        "experiment,seed,family,variant,parameters,best_epoch,"
        "component_rmse_A,component_mae_A,vector_error_mean_A,"
        "trajectory_rmse_mean_A,trajectory_rmse_std_A,"
        "so3_equiv_rmse_A,so3_degradation,"
        "improper_equiv_rmse_A\n"
    )

    for result in all_results:
        stress = find_stress(
            result["experiment_name"],
            result["seed"],
        )

        if stress is None:
            so3_eq = ""
            so3_deg = ""
            improper_eq = ""
        else:
            so3_eq = (
                stress[
                    "proper_SO3"
                ][
                    "equivariance_rmse_mean"
                ]
            )
            so3_deg = (
                stress[
                    "proper_SO3"
                ][
                    "degradation_mean"
                ]
            )
            improper_eq = (
                stress[
                    "improper_O3"
                ][
                    "equivariance_rmse_mean"
                ]
            )

        m = result["validation"]

        f.write(
            f"{result['experiment_name']},"
            f"{result['seed']},"
            f"{result['family']},"
            f"{result['variant']},"
            f"{result['parameters']},"
            f"{result['best_epoch']},"
            f"{m['component_rmse']},"
            f"{m['component_mae']},"
            f"{m['vector_error_mean']},"
            f"{m['trajectory_component_rmse_mean']},"
            f"{m['trajectory_component_rmse_std']},"
            f"{so3_eq},"
            f"{so3_deg},"
            f"{improper_eq}\n"
        )


# 
# Archive outputs
# 

archive_path = shutil.make_archive(
    OUTPUT_DIR,
    "zip",
    OUTPUT_DIR,
)

print()
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)
print("Total time:", f"{total_minutes:.2f} min")
print("Results JSON:", RESULT_FILE)
print("Summary CSV:", csv_file)
print("Archive:", archive_path)
print()
print("TEST SET HAS NOT BEEN LOADED OR EVALUATED.")
