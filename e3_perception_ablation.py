#!/usr/bin/env python3
"""E3 geometric-perception ablation runner for the anorthite dynamics dataset.

Runs every Part-I ablation that can be defined before seeing ablation results.
Excluded by design: winner-dependent interaction sweeps, B-state reruns of
selected winners, and learned-controller training.

TEST IS NEVER RESOLVED OR LOADED.

Fixed model seeds for every configuration:
    9078, 4577, 3320, 3733, 9428
"""
from __future__ import annotations

import csv, glob, importlib, importlib.util, itertools, json, math, os
from pathlib import Path
import platform, random, re, subprocess, sys, time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, replace
from typing import Optional, Sequence, Dict, List


def _ensure_package(import_name, pip_spec, required_version=None):
    missing = importlib.util.find_spec(import_name) is None
    if not missing and required_version is not None:
        mod = importlib.import_module(import_name)
        missing = getattr(mod, "__version__", None) != required_version
    if missing:
        print(f"Installing {pip_spec} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_spec])
        importlib.invalidate_caches()


_ensure_package("e3nn", "e3nn==0.6.0", "0.6.0")
_ensure_package("torch_geometric", "torch-geometric")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import e3nn
from e3nn import o3

# -----------------------------------------------------------------------------
# Experimental contract
# -----------------------------------------------------------------------------
FIXED_TRAIN_SEEDS = [9078, 4577, 3320, 3733, 9428]
ROTATION_SEED = 8675309

MD_TIMESTEP_PS = 0.001
EXPECTED_FRAME_STRIDE = 10
ATOMIC_NUMBER_BY_RAW_TYPE = {1: 20, 2: 12, 3: 13, 4: 14, 5: 8}
ELEMENT_BY_Z = {8: "O", 12: "Mg", 13: "Al", 14: "Si", 20: "Ca"}
MAX_ATOMIC_NUMBER = 118

BASELINE_CUTOFF = 3.5
BASELINE_SPECIES_EMBED_DIM = 16
BASELINE_HIDDEN_IRREPS = "32x0e + 16x1o"
BASELINE_EDGE_LMAX = 2
BASELINE_NUM_RADIAL = 8
BASELINE_RADIAL_HIDDEN = 64
BASELINE_DEPTH = 2
BASELINE_AGGREGATION = "mean"
OUTPUT_IRREPS = "1x1o"
EXPECTED_BASELINE_C_PARAMS = 336480

BATCH_SIZE = 48
FULL_MAX_EPOCHS = 400
SCREEN_MAX_EPOCHS = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
VAL_EVERY = 5
EARLY_STOP_PATIENCE_EPOCHS = 60
LR_PATIENCE_EPOCHS = 15
LR_FACTOR = 0.5
MIN_LR = 1e-6
GRAD_CLIP_NORM = 10.0

EQ_SELFTEST_RMSE_TOL = 5e-5
EQ_SELFTEST_MAX_TOL = 2e-4
SELFTEST_GRAPHS = 8
DEFAULT_SYMMETRY_GRAPHS = 512
DEFAULT_SYMMETRY_SO3 = 1
DEFAULT_RANDOM_IMPROPER = 1
ORACLE_COST_LAMBDAS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]


def default_output_dir():
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/e3_perception_ablation_outputs"
    if os.path.isdir("/content"):
        return "/content/e3_perception_ablation_outputs"
    return os.path.abspath("./e3_perception_ablation_outputs")


def discover_root():
    candidates = [
        os.environ.get("ABLATION_ROOT"),
        "/content/processed_production",
        "/kaggle/working/processed_production",
        os.path.abspath("./processed_production"),
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "splits.json")):
            return c
    if os.path.isdir("/kaggle/input"):
        found = glob.glob("/kaggle/input/**/processed_production/splits.json", recursive=True)
        if found:
            return os.path.dirname(found[0])
    return None


# -----------------------------------------------------------------------------
# KAGGLE / NOTEBOOK RUN SETTINGS
# -----------------------------------------------------------------------------
# This script is intentionally zero-argument. Edit these values here if needed,
# then press Run in Kaggle. No terminal flags are required.
#
# ROOT_OVERRIDE:
#   Leave as None to auto-discover a folder containing processed_production/
#   and splits.json under /kaggle/input, /kaggle/working, or /content.
#   If auto-discovery ever fails, set the exact path manually, e.g.
#   ROOT_OVERRIDE = "/kaggle/input/my-dataset/processed_production"
ROOT_OVERRIDE = None

# Output goes to /kaggle/working/e3_perception_ablation_outputs on Kaggle.
OUTPUT_DIR_OVERRIDE = None

# Real Part-I run settings.
RUN_MODE = "full"                    # "full" or "screen"
FAMILIES_TO_RUN = "all"             # "all" or comma-separated family names
CONFIGS_TO_RUN = None                # None or comma-separated config names
SEED_LIMIT = None                    # None = all five fixed seeds
SOURCE_GRAPH_CUTOFF = BASELINE_CUTOFF  # current stored graph is 3.5 A
SYMMETRY_GRAPHS = DEFAULT_SYMMETRY_GRAPHS
SYMMETRY_SO3 = DEFAULT_SYMMETRY_SO3
RANDOM_IMPROPER = DEFAULT_RANDOM_IMPROPER
SKIP_SYMMETRY = False
LIST_CONFIGS_ONLY = False
DRY_RUN = False
RESUME = True
FORCE_RERUN = False
RUN_BATCH_SIZE = BATCH_SIZE


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def safe_name(s):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_").lower()


def json_dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def synchronize():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def random_so3_matrix(rng):
    A = rng.normal(size=(3, 3)); Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R)); signs[signs == 0] = 1.0
    Q = Q @ np.diag(signs)
    if np.linalg.det(Q) < 0: Q[:, 0] *= -1.0
    return Q.astype(np.float32)


# -----------------------------------------------------------------------------
# Experiment manifest
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    family: str
    cutoff: float = BASELINE_CUTOFF
    edge_lmax: int = BASELINE_EDGE_LMAX
    hidden_irreps: str = BASELINE_HIDDEN_IRREPS
    num_radial: int = BASELINE_NUM_RADIAL
    radial_hidden: int = BASELINE_RADIAL_HIDDEN
    depth: int = BASELINE_DEPTH
    aggregation: str = BASELINE_AGGREGATION
    species_embed_dim: int = BASELINE_SPECIES_EMBED_DIM
    variant: str = "C"
    notes: str = ""
    deferred_reason: Optional[str] = None

    def validate(self):
        assert self.variant == "C"
        assert self.cutoff > 0 and self.edge_lmax >= 0 and self.num_radial > 0 and self.depth > 0
        assert self.aggregation in {"mean", "sum", "global_sqrt_degree", "local_sqrt_degree", "learned_gate"}
        o3.Irreps(self.hidden_irreps)


BASELINE_CONFIG = ExperimentConfig(
    "baseline_v2", "baseline",
    notes="Architecture-identical E3-C baseline with V2 training protocol."
)


def base_copy(name, family, **kwargs):
    return replace(BASELINE_CONFIG, name=name, family=family, **kwargs)


def initial_manifest(source_graph_cutoff):
    cfgs = [BASELINE_CONFIG]
    cfgs += [
        base_copy("agg_sum", "aggregation", aggregation="sum"),
        base_copy("agg_global_sqrt_degree", "aggregation", aggregation="global_sqrt_degree"),
        base_copy("agg_local_sqrt_degree", "aggregation", aggregation="local_sqrt_degree"),
        base_copy("agg_learned_invariant_gate", "aggregation", aggregation="learned_gate",
                  notes="Invariant state+radial edge gate; weighted-mean normalization."),
    ]
    # l_edge=3/4 against the baseline 0e+1o hidden state would partly expose
    # channels with no allowed path back into the retained representation, so
    # the clean low-order edge sweep stops at l=2 (baseline).
    cfgs += [
        base_copy("edge_lmax0", "edge_angular", edge_lmax=0),
        base_copy("edge_lmax1", "edge_angular", edge_lmax=1),
    ]
    hidden_l2 = "32x0e + 16x1o + 8x2e"
    hidden_l3 = "32x0e + 16x1o + 8x2e + 4x3o"
    cfgs += [
        base_copy("hidden_add_l2", "hidden_angular", hidden_irreps=hidden_l2),
        base_copy("hidden_add_l2_l3", "hidden_angular", hidden_irreps=hidden_l3),
        base_copy("angular_bandwidth_l3", "hidden_angular", hidden_irreps=hidden_l3, edge_lmax=3),
        base_copy("angular_bandwidth_l4", "hidden_angular", hidden_irreps=hidden_l3, edge_lmax=4),
    ]
    for rc in [2.5, 3.0, 4.0, 4.5, 5.0]:
        deferred = None
        if rc > source_graph_cutoff + 1e-8:
            deferred = (f"cutoff {rc:.1f} A exceeds stored/source graph cutoff "
                        f"{source_graph_cutoff:.1f} A; regenerate a superset graph first")
        cfgs.append(base_copy(f"cutoff_{str(rc).replace('.', 'p')}A", "radial_extent",
                              cutoff=rc, deferred_reason=deferred))
    for n in [4, 16, 32]: cfgs.append(base_copy(f"rbf_{n}", "radial_resolution", num_radial=n))
    for d in [1, 3, 4, 5]: cfgs.append(base_copy(f"depth_{d}", "depth", depth=d))
    cfgs += [
        base_copy("width_half_multiplicity", "width", hidden_irreps="16x0e + 8x1o"),
        base_copy("width_double_multiplicity", "width", hidden_irreps="64x0e + 32x1o"),
    ]
    for c in cfgs: c.validate()
    assert len({c.name for c in cfgs}) == len(cfgs)
    return cfgs


# -----------------------------------------------------------------------------
# Split / cache
# -----------------------------------------------------------------------------
def load_split_manifest(root):
    with open(os.path.join(root, "splits.json"), "r", encoding="utf-8") as f:
        s = json.load(f)
    train, val, test = s["splits"]["train"], s["splits"]["val"], s["splits"]["test"]
    a, b, c = {x["run_id"] for x in train}, {x["run_id"] for x in val}, {x["run_id"] for x in test}
    if a & b or a & c or b & c: raise RuntimeError("Split manifest is not trajectory-disjoint")
    return s, train, val, test


def get_sample_records(root, entries):
    out = []
    for traj_idx, entry in enumerate(entries):
        sample_dir = os.path.join(root, entry["sample_dir"].replace("\\", "/"))
        if not os.path.isdir(sample_dir): raise FileNotFoundError(sample_dir)
        files = sorted(os.path.join(sample_dir, f) for f in os.listdir(sample_dir)
                       if f.startswith("sample_") and f.endswith(".npz"))
        expected = int(entry["num_samples"])
        if len(files) != expected: raise RuntimeError(f"{entry['run_id']}: expected {expected}, got {len(files)}")
        for sample_idx, path in enumerate(files):
            out.append(dict(path=path, traj_idx=traj_idx, run_id=entry["run_id"],
                            sample_idx=sample_idx, num_samples_in_trajectory=expected))
    return out


class CachedGraph:
    __slots__ = ("atomic_numbers", "raw_types", "atom_ids", "velocity", "force",
                 "edge_index", "edge_vector", "edge_distance", "y", "traj_idx",
                 "run_id", "sample_idx", "num_samples_in_trajectory")
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)


def raw_types_to_atomic_numbers(raw_types):
    z = np.empty_like(raw_types, dtype=np.int64)
    for rt in np.unique(raw_types):
        if int(rt) not in ATOMIC_NUMBER_BY_RAW_TYPE: raise RuntimeError(f"Unknown LAMMPS type {rt}")
        z[raw_types == rt] = ATOMIC_NUMBER_BY_RAW_TYPE[int(rt)]
    return z


def first_key(d, keys):
    return next((k for k in keys if k in d), None)


def load_cache(records, collect_training_stats, label):
    cache, seen_species, frame_strides = [], set(), set()
    vel_sq_total = force_sq_total = 0.0; components = 0; atoms = 0
    vel_sum = np.zeros(3); force_sum = np.zeros(3)
    max_edge, min_edge, total_edges = 0.0, float("inf"), 0
    t0 = time.time()
    for i, r in enumerate(records):
        with np.load(r["path"]) as d:
            raw = d["atom_types"].astype(np.int64); z = raw_types_to_atomic_numbers(raw); n = len(raw)
            kid = first_key(d, ["atom_ids", "atom_id", "ids", "id"])
            atom_ids = np.arange(1, n + 1, dtype=np.int64) if kid is None else np.asarray(d[kid], dtype=np.int64).reshape(-1)
            v = np.stack([d["vx_t"], d["vy_t"], d["vz_t"]], 1).astype(np.float32)
            f = np.stack([d["fx_t"], d["fy_t"], d["fz_t"]], 1).astype(np.float32)
            ei = np.stack([d["edge_src"].astype(np.int64), d["edge_dst"].astype(np.int64)], 0)
            ev = d["edge_vector"].astype(np.float32); ed = d["edge_distance"].astype(np.float32).reshape(-1)
            y = d["non_affine_displacement"].astype(np.float32)
            if "timestep_t" in d and "timestep_t1" in d:
                frame_strides.add(int(np.asarray(d["timestep_t1"]).item()) - int(np.asarray(d["timestep_t"]).item()))
        if v.shape != (n, 3) or f.shape != (n, 3) or y.shape != (n, 3): raise RuntimeError(f"Bad vectors: {r['path']}")
        if ei.shape[0] != 2 or ev.shape != (ei.shape[1], 3) or len(ed) != ei.shape[1]: raise RuntimeError(f"Bad edges: {r['path']}")
        if not all(np.all(np.isfinite(a)) for a in [v, f, ev, ed, y]): raise RuntimeError(f"Non-finite: {r['path']}")
        if np.any(ed <= 0): raise RuntimeError(f"Non-positive distance: {r['path']}")
        if ei.size and (ei.min() < 0 or ei.max() >= n): raise RuntimeError(f"Bad edge index: {r['path']}")
        if np.max(np.abs(np.linalg.norm(ev, axis=1) - ed)) > 2e-4: raise RuntimeError(f"Edge geometry mismatch: {r['path']}")
        seen_species.update(map(int, np.unique(z))); max_edge = max(max_edge, float(ed.max())); min_edge = min(min_edge, float(ed.min())); total_edges += len(ed)
        if collect_training_stats:
            vv, ff = v.astype(np.float64), f.astype(np.float64)
            vel_sq_total += float(np.sum(vv ** 2)); force_sq_total += float(np.sum(ff ** 2)); components += vv.size
            vel_sum += vv.sum(0); force_sum += ff.sum(0); atoms += n
        cache.append(CachedGraph(
            atomic_numbers=torch.from_numpy(z), raw_types=torch.from_numpy(raw), atom_ids=torch.from_numpy(atom_ids),
            velocity=torch.from_numpy(v), force=torch.from_numpy(f), edge_index=torch.from_numpy(ei),
            edge_vector=torch.from_numpy(ev), edge_distance=torch.from_numpy(ed), y=torch.from_numpy(y),
            traj_idx=int(r["traj_idx"]), run_id=r["run_id"], sample_idx=int(r["sample_idx"]),
            num_samples_in_trajectory=int(r["num_samples_in_trajectory"])))
        if (i + 1) % 1000 == 0: print(f"{label}: {i+1}/{len(records)}")
    stats = dict(seen_species=sorted(seen_species), frame_strides=sorted(frame_strides),
                 max_stored_edge_distance=max_edge, min_stored_edge_distance=min_edge,
                 mean_edges_per_graph=total_edges / len(cache), load_seconds=time.time() - t0)
    if collect_training_stats:
        stats.update(velocity_scale=math.sqrt(vel_sq_total/components), force_scale=math.sqrt(force_sq_total/components),
                     velocity_mean_diagnostic=(vel_sum/atoms).tolist(), force_mean_diagnostic=(force_sum/atoms).tolist())
    print(f"{label} loaded in {stats['load_seconds']:.1f}s")
    return cache, stats


class GraphViewDataset(torch.utils.data.Dataset):
    def __init__(self, cache, cutoff, velocity_scale, force_scale, transform_matrix=None):
        self.cache = cache; self.cutoff = float(cutoff); self.velocity_scale = velocity_scale; self.force_scale = force_scale
        self.keep_indices, self.edge_counts = [], []
        for g in cache:
            keep = torch.nonzero(g.edge_distance <= cutoff + 1e-7, as_tuple=False).flatten()
            if keep.numel() == 0: raise RuntimeError(f"Zero-edge graph at cutoff {cutoff}")
            self.keep_indices.append(keep); self.edge_counts.append(int(keep.numel()))
        if transform_matrix is None: self.Q = None
        else:
            Q = np.asarray(transform_matrix, dtype=np.float32)
            if Q.shape != (3,3) or np.max(np.abs(Q @ Q.T - np.eye(3))) > 1e-5: raise ValueError("Invalid O(3) transform")
            self.Q = torch.tensor(Q, dtype=torch.float32)
    def __len__(self): return len(self.cache)
    def __getitem__(self, idx):
        g, keep = self.cache[idx], self.keep_indices[idx]
        v, f, ev, y = g.velocity, g.force, g.edge_vector[keep], g.y
        if self.Q is not None:
            QT = self.Q.T; v = v @ QT; f = f @ QT; ev = ev @ QT; y = y @ QT
        return Data(atomic_numbers=g.atomic_numbers, atom_ids=g.atom_ids,
                    velocity=v/self.velocity_scale, force=f/self.force_scale,
                    edge_index=g.edge_index[:, keep], edge_vector=ev, edge_distance=g.edge_distance[keep], y=y,
                    traj_index=torch.tensor([g.traj_idx], dtype=torch.long),
                    cache_index=torch.tensor([idx], dtype=torch.long), num_nodes=len(g.atomic_numbers))
    @property
    def mean_edges(self): return float(np.mean(self.edge_counts))
    @property
    def p95_edges(self): return float(np.percentile(self.edge_counts, 95))
    @property
    def global_mean_degree(self):
        return float(sum(self.edge_counts) / sum(len(g.atomic_numbers) for g in self.cache))


def make_loader(ds, shuffle, seed, batch_size):
    gen = torch.Generator(); gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen if shuffle else None,
                      num_workers=0, pin_memory=torch.cuda.is_available())

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def basic_metric_dict(pred, target):
    e = pred.float() - target.float(); mse = torch.mean(e**2).item(); vec = torch.linalg.vector_norm(e, dim=-1)
    return dict(component_mse=float(mse), component_rmse=float(math.sqrt(mse)),
                component_mae=float(torch.mean(torch.abs(e)).item()),
                vector_error_mean=float(vec.mean().item()),
                vector_error_rms=float(torch.sqrt(torch.mean(vec**2)).item()),
                vector_error_median=float(torch.quantile(vec, .50).item()),
                vector_error_p95=float(torch.quantile(vec, .95).item()), vector_error_max=float(vec.max().item()))


def detailed_metric_dict(pred, target, trajectory_index, atomic_numbers, trajectory_names):
    m = basic_metric_dict(pred, target); e = pred.float() - target.float(); per_t = {}; rmses = []
    for ti in torch.unique(trajectory_index).tolist():
        mask = trajectory_index == ti; ee = e[mask]; vec = torch.linalg.vector_norm(ee, dim=-1); rmse = float(torch.sqrt(torch.mean(ee**2)).item())
        name = trajectory_names[int(ti)]; rmses.append(rmse)
        per_t[name] = dict(component_rmse=rmse, component_mae=float(torch.mean(torch.abs(ee)).item()),
                           vector_error_mean=float(vec.mean().item()), vector_error_p95=float(torch.quantile(vec,.95).item()),
                           n_atom_instances=int(mask.sum().item()))
    rr = np.asarray(rmses, dtype=np.float64)
    m.update(trajectory_component_rmse_mean=float(rr.mean()), trajectory_component_rmse_std=float(rr.std(ddof=1) if len(rr)>1 else 0),
             trajectory_count=len(rr), per_trajectory=per_t)
    per_s = {}
    for z in torch.unique(atomic_numbers).tolist():
        mask = atomic_numbers == z; ee = e[mask]; vec = torch.linalg.vector_norm(ee, dim=-1); label = ELEMENT_BY_Z.get(int(z), f"Z{z}")
        per_s[label] = dict(Z=int(z), component_rmse=float(torch.sqrt(torch.mean(ee**2)).item()),
                            component_mae=float(torch.mean(torch.abs(ee)).item()), vector_error_mean=float(vec.mean().item()),
                            vector_error_p95=float(torch.quantile(vec,.95).item()), n_atom_instances=int(mask.sum().item()))
    m["per_species"] = per_s
    return m


# -----------------------------------------------------------------------------
# E3 model
# -----------------------------------------------------------------------------
def radial_basis(distance, cutoff, num_basis):
    x = distance/cutoff; centers = torch.linspace(0.,1.,num_basis,device=distance.device,dtype=distance.dtype); width=1./num_basis
    basis = torch.exp(-.5*((x[:,None]-centers[None,:])/width)**2)
    env = .5*(torch.cos(math.pi*torch.clamp(x,0.,1.))+1.); env = env*(x<=1.).to(distance.dtype)
    return basis*env[:,None]


def invariant_summary(x, irreps):
    pieces=[]; off=0
    for mul, ir in irreps:
        dim=mul*ir.dim; b=x[:,off:off+dim].reshape(x.shape[0],mul,ir.dim)
        pieces.append(b[...,0] if ir.l==0 else torch.sqrt(torch.sum(b**2,dim=-1)+1e-12)); off += dim
    return torch.cat(pieces, dim=-1)


def invariant_summary_dim(irreps): return int(sum(mul for mul,_ in irreps))


class EquivariantConv(nn.Module):
    def __init__(self, irreps_in, irreps_out, cfg, global_mean_degree):
        super().__init__(); self.irreps_in=o3.Irreps(irreps_in); self.irreps_out=o3.Irreps(irreps_out)
        self.irreps_edge=o3.Irreps.spherical_harmonics(lmax=cfg.edge_lmax); self.cutoff=cfg.cutoff; self.num_radial=cfg.num_radial
        self.aggregation=cfg.aggregation; self.global_mean_degree=float(global_mean_degree)
        self.tp=o3.FullyConnectedTensorProduct(self.irreps_in,self.irreps_edge,self.irreps_out,shared_weights=False,internal_weights=False)
        if self.tp.weight_numel == 0: raise RuntimeError(f"No tensor-product paths: {self.irreps_in} x {self.irreps_edge} -> {self.irreps_out}")
        self.radial_mlp=nn.Sequential(nn.Linear(cfg.num_radial,cfg.radial_hidden),nn.SiLU(),nn.Linear(cfg.radial_hidden,self.tp.weight_numel))
        if cfg.aggregation == "learned_gate":
            d=2*invariant_summary_dim(self.irreps_in)+cfg.num_radial; h=max(32,min(128,2*d))
            self.edge_gate=nn.Sequential(nn.Linear(d,h),nn.SiLU(),nn.Linear(h,1))
        else: self.edge_gate=None
    def forward(self,x,edge_index,edge_vector,edge_distance):
        src,dst=edge_index[0],edge_index[1]
        sh=o3.spherical_harmonics(self.irreps_edge,edge_vector,normalize=True,normalization="component")
        radial=radial_basis(edge_distance,self.cutoff,self.num_radial); weights=self.radial_mlp(radial)
        msg=self.tp(x[src],sh,weights); gate=None
        if self.edge_gate is not None:
            inv=invariant_summary(x,self.irreps_in); gate=torch.sigmoid(self.edge_gate(torch.cat([inv[src],inv[dst],radial],-1))); msg=msg*gate
        out=torch.zeros((x.shape[0],self.irreps_out.dim),dtype=x.dtype,device=x.device); out.index_add_(0,dst,msg)
        degree=torch.zeros(x.shape[0],dtype=x.dtype,device=x.device); degree.index_add_(0,dst,torch.ones(dst.shape[0],dtype=x.dtype,device=x.device))
        if self.aggregation=="mean": out=out/degree.clamp(min=1.)[:,None]
        elif self.aggregation=="sum": pass
        elif self.aggregation=="global_sqrt_degree": out=out/math.sqrt(max(self.global_mean_degree,1e-8))
        elif self.aggregation=="local_sqrt_degree": out=out/torch.sqrt(degree.clamp(min=1.))[:,None]
        elif self.aggregation=="learned_gate":
            den=torch.zeros(x.shape[0],dtype=x.dtype,device=x.device); den.index_add_(0,dst,gate[:,0]); out=out/den.clamp(min=1e-6)[:,None]
        else: raise ValueError(self.aggregation)
        return out


class EquivariantActivation(nn.Module):
    def __init__(self, irreps):
        super().__init__(); self.irreps=o3.Irreps(irreps); self.tensor_gates=nn.ParameterList(); self.tensor_biases=nn.ParameterList()
        for mul,ir in self.irreps:
            if ir.l>0: self.tensor_gates.append(nn.Parameter(torch.ones(mul))); self.tensor_biases.append(nn.Parameter(torch.zeros(mul)))
    def forward(self,x):
        pieces=[]; off=0; ti=0
        for mul,ir in self.irreps:
            dim=mul*ir.dim; b=x[:,off:off+dim].reshape(x.shape[0],mul,ir.dim)
            if ir.l==0: b=F.silu(b)
            else:
                norms=torch.sqrt(torch.sum(b**2,dim=-1)+1e-12); g=torch.sigmoid(norms*self.tensor_gates[ti][None,:]+self.tensor_biases[ti][None,:]); b=b*g[:,:,None]; ti+=1
            pieces.append(b.reshape(x.shape[0],dim)); off += dim
        return torch.cat(pieces,-1)


class E3PerceptionGNN(nn.Module):
    def __init__(self,cfg,global_mean_degree):
        super().__init__(); self.cfg=cfg; self.species_embedding=nn.Embedding(MAX_ATOMIC_NUMBER+1,cfg.species_embed_dim)
        self.irreps_input=o3.Irreps(f"{cfg.species_embed_dim}x0e + 2x1o"); self.irreps_hidden=o3.Irreps(cfg.hidden_irreps); self.irreps_output=o3.Irreps(OUTPUT_IRREPS)
        if not any(ir.l==1 and ir.p==-1 and mul>0 for mul,ir in self.irreps_hidden): raise ValueError("Hidden state requires a 1o channel for displacement head")
        self.input_linear=o3.Linear(self.irreps_input,self.irreps_hidden)
        self.convs=nn.ModuleList([EquivariantConv(self.irreps_hidden,self.irreps_hidden,cfg,global_mean_degree) for _ in range(cfg.depth)])
        self.acts=nn.ModuleList([EquivariantActivation(self.irreps_hidden) for _ in range(cfg.depth)])
        self.output_linear=o3.Linear(self.irreps_hidden,self.irreps_output)
    def forward(self,data):
        species=self.species_embedding(data.atomic_numbers); x=self.input_linear(torch.cat([species,data.velocity,data.force],-1))
        for conv,act in zip(self.convs,self.acts): x=act(x+conv(x,data.edge_index,data.edge_vector,data.edge_distance))
        return self.output_linear(x)
    @property
    def tp_weight_numel_per_layer(self): return [int(c.tp.weight_numel) for c in self.convs]


def model_param_count_for_config(cfg,degree): return parameter_count(E3PerceptionGNN(cfg,degree))


def add_parameter_matched_controls(cfgs,degree):
    by={c.name:c for c in cfgs}; used={c.hidden_irreps for c in cfgs}; candidates=[]
    for m0 in range(24,81,4):
        for m1 in range(12,41,2):
            ir=f"{m0}x0e + {m1}x1o"
            if ir in used: continue
            c=base_copy("_tmp","hidden_angular",hidden_irreps=ir)
            try: p=model_param_count_for_config(c,degree)
            except Exception: continue
            candidates.append((p,ir))
    extra=[]
    for target_name,suffix in [("hidden_add_l2","l2"),("hidden_add_l2_l3","l2_l3")]:
        tp=model_param_count_for_config(by[target_name],degree); ranked=sorted((abs(p-tp),p,ir) for p,ir in candidates)
        if ranked:
            diff,p,ir=ranked[0]; extra.append(base_copy(f"hidden_lowL_parammatch_{suffix}","hidden_angular",hidden_irreps=ir,
                notes=f"0e+1o parameter control: target={tp}, control={p}, absdiff={diff}")); candidates=[x for x in candidates if x[1]!=ir]
    return cfgs+extra


# -----------------------------------------------------------------------------
# Prediction / equivariance
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_dataset(model,ds,device,batch_size):
    loader=make_loader(ds,False,0,batch_size); model.eval(); pp=[];tt=[];tr=[];zz=[];ci=[]; synchronize(); t0=time.perf_counter()
    for b in loader:
        b=b.to(device,non_blocking=True); p=model(b); pp.append(p.float().cpu()); tt.append(b.y.float().cpu()); tr.append(b.traj_index[b.batch].long().cpu()); zz.append(b.atomic_numbers.long().cpu()); ci.append(b.cache_index.view(-1)[b.batch].long().cpu())
    synchronize(); dt=time.perf_counter()-t0
    return torch.cat(pp),torch.cat(tt),torch.cat(tr),torch.cat(zz),torch.cat(ci),dt


@torch.no_grad()
def evaluate_basic(model,loader,device):
    model.eval(); ss=sa=0.; n=0
    for b in loader:
        b=b.to(device,non_blocking=True); e=model(b).float()-b.y.float(); ss += torch.sum(e**2).item(); sa += torch.sum(torch.abs(e)).item(); n += e.numel()
    mse=ss/n; return float(mse),float(math.sqrt(mse)),float(sa/n)


def make_stress_transforms(n_so3,n_random_improper):
    rng=np.random.default_rng(ROTATION_SEED); proper=[dict(name=f"SO3_{i+1:02d}",kind="proper",matrix=random_so3_matrix(rng)) for i in range(n_so3)]
    rx=np.diag([-1.,1.,1.]).astype(np.float32); imp=[dict(name="reflect_x",kind="improper",matrix=rx),dict(name="inversion",kind="improper",matrix=-np.eye(3,dtype=np.float32))]
    for i in range(n_random_improper): imp.append(dict(name=f"improper_random_{i+1:02d}",kind="improper",matrix=(random_so3_matrix(rng)@rx).astype(np.float32)))
    return proper+imp


@torch.no_grad()
def equivariance_self_test(model,cache,cfg,vs,fs,device,batch_size):
    tiny=cache[:min(SELFTEST_GRAPHS,len(cache))]; base=GraphViewDataset(tiny,cfg.cutoff,vs,fs); p0,*_=predict_dataset(model,base,device,min(batch_size,len(tiny)))
    rng=np.random.default_rng(ROTATION_SEED); tests=[("proper_rotation",random_so3_matrix(rng)),("improper_reflection",np.diag([-1.,1.,1.]).astype(np.float32))]; out={}
    for name,Q in tests:
        p,*_=predict_dataset(model,GraphViewDataset(tiny,cfg.cutoff,vs,fs,Q),device,min(batch_size,len(tiny))); expected=p0@torch.tensor(Q,dtype=torch.float32).T; diff=p-expected
        rmse=float(torch.sqrt(torch.mean(diff**2)).item()); mx=float(torch.max(torch.abs(diff)).item()); out[name]=dict(rmse=rmse,max_abs=mx,determinant=float(np.linalg.det(Q)))
        if rmse>EQ_SELFTEST_RMSE_TOL or mx>EQ_SELFTEST_MAX_TOL: raise RuntimeError(f"Equivariance preflight failed {cfg.name}/{name}: {rmse:.3e}, {mx:.3e}")
    return out


@torch.no_grad()
def symmetry_stress_test(model,cache,cfg,vs,fs,device,batch_size,n_graphs,transforms):
    sub=cache[:min(n_graphs,len(cache))]; base=GraphViewDataset(sub,cfg.cutoff,vs,fs); p0,t0,*_=predict_dataset(model,base,device,batch_size); rm0=basic_metric_dict(p0,t0)["component_rmse"]; rows=[]
    for item in transforms:
        Q=item["matrix"]; p,t,*_=predict_dataset(model,GraphViewDataset(sub,cfg.cutoff,vs,fs,Q),device,batch_size); acc=basic_metric_dict(p,t); eq=basic_metric_dict(p,p0@torch.tensor(Q,dtype=torch.float32).T)
        rows.append(dict(name=item["name"],kind=item["kind"],determinant=float(np.linalg.det(Q)),target_component_rmse=acc["component_rmse"],equivariance_component_rmse=eq["component_rmse"],equivariance_component_mae=eq["component_mae"],accuracy_degradation_factor=float(acc["component_rmse"]/max(rm0,1e-30))))
    return dict(base_component_rmse=rm0,n_graphs=len(sub),transforms=rows)

# -----------------------------------------------------------------------------
# Shared validation-state descriptors for controller/oracle analysis
# -----------------------------------------------------------------------------
def parse_run_descriptor(run_id):
    m=re.match(r"(?P<mode>[^_]+)_strain(?P<strain>\d+)_seed(?P<seed>\d+)",run_id)
    if not m: return "unknown",float("nan"),-1
    return m.group("mode"),float(m.group("strain"))/10.,int(m.group("seed"))


def neighbor_descriptors(g,cutoff):
    keep=g.edge_distance<=cutoff+1e-7; dst=g.edge_index[1,keep].numpy(); dist=g.edge_distance[keep].numpy().astype(np.float64); n=len(g.atomic_numbers)
    degree=np.zeros(n,np.int32); s=np.zeros(n); s2=np.zeros(n); mn=np.full(n,np.inf)
    np.add.at(degree,dst,1); np.add.at(s,dst,dist); np.add.at(s2,dst,dist**2); np.minimum.at(mn,dst,dist)
    den=np.maximum(degree,1); mean=s/den; std=np.sqrt(np.maximum(s2/den-mean**2,0)); mn[~np.isfinite(mn)]=np.nan
    density=degree/((4/3)*math.pi*cutoff**3)
    return degree,mean,std,mn,density


def build_state_descriptors(val_cache,cutoff,path):
    arr=defaultdict(list); run_names=[]; run_to_i={}; sid=0; mode_map={"iso":0,"x":1,"y":2,"z":3}
    for cache_idx,g in enumerate(val_cache):
        n=len(g.atomic_numbers); mode,strain,mdseed=parse_run_descriptor(g.run_id)
        if g.run_id not in run_to_i: run_to_i[g.run_id]=len(run_names); run_names.append(g.run_id)
        v=g.velocity.numpy().astype(np.float64); f=g.force.numpy().astype(np.float64); y=g.y.numpy().astype(np.float64); deg,mean,std,mn,dens=neighbor_descriptors(g,cutoff)
        progress=g.sample_idx/max(1,g.num_samples_in_trajectory-1)
        def add(k,x): arr[k].append(x)
        add("state_id",np.arange(sid,sid+n,dtype=np.int64)); add("cache_index",np.full(n,cache_idx,np.int32)); add("trajectory_index",np.full(n,g.traj_idx,np.int16)); add("run_name_index",np.full(n,run_to_i[g.run_id],np.int16)); add("sample_index",np.full(n,g.sample_idx,np.int32)); add("atom_id",g.atom_ids.numpy().astype(np.int32)); add("Z",g.atomic_numbers.numpy().astype(np.int16))
        add("v_norm",np.linalg.norm(v,axis=1).astype(np.float32)); add("f_norm",np.linalg.norm(f,axis=1).astype(np.float32)); add("v_dot_f",np.sum(v*f,axis=1).astype(np.float32))
        add("coordination",deg.astype(np.int16)); add("mean_neighbor_distance",mean.astype(np.float32)); add("std_neighbor_distance",std.astype(np.float32)); add("min_neighbor_distance",mn.astype(np.float32)); add("local_density",dens.astype(np.float32))
        add("strain_percent",np.full(n,strain,np.float32)); add("md_seed",np.full(n,mdseed,np.int32)); add("loading_progress",np.full(n,progress,np.float32)); add("loading_mode_code",np.full(n,mode_map.get(mode,-1),np.int8))
        add("target_dx",y[:,0].astype(np.float32)); add("target_dy",y[:,1].astype(np.float32)); add("target_dz",y[:,2].astype(np.float32)); add("target_norm",np.linalg.norm(y,axis=1).astype(np.float32)); sid += n
    out={k:np.concatenate(v) for k,v in arr.items()}; out["run_names"]=np.asarray(run_names,dtype="U128"); out["loading_mode_names"]=np.asarray(["iso","x","y","z"],dtype="U16")
    np.savez_compressed(path,**out); return dict(n_states=sid,n_trajectories=len(run_names),path=path)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def training_protocol(mode):
    mx=FULL_MAX_EPOCHS if mode=="full" else SCREEN_MAX_EPOCHS
    return dict(max_epochs=mx,learning_rate=LEARNING_RATE,weight_decay=WEIGHT_DECAY,val_every=VAL_EVERY,
                early_stop_patience_epochs=EARLY_STOP_PATIENCE_EPOCHS,early_stop_patience_evals=max(1,math.ceil(EARLY_STOP_PATIENCE_EPOCHS/VAL_EVERY)),
                lr_patience_epochs=LR_PATIENCE_EPOCHS,lr_patience_evals=max(1,math.ceil(LR_PATIENCE_EPOCHS/VAL_EVERY)),lr_factor=LR_FACTOR,min_lr=MIN_LR,grad_clip_norm=GRAD_CLIP_NORM)


def save_history_csv(history,path):
    if not history:return
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)


def train_one(cfg,seed,train_cache,val_cache,train_ds,val_ds,vs,fs,val_names,output_dir,device,batch_size,protocol,transforms,symmetry_graphs,skip_symmetry,resume,force_rerun):
    run_dir=os.path.join(output_dir,"runs",safe_name(cfg.name),f"seed_{seed}"); os.makedirs(run_dir,exist_ok=True)
    result_path=os.path.join(run_dir,"result.json"); best_path=os.path.join(run_dir,"best.pt"); last_path=os.path.join(run_dir,"last.pt"); hist_path=os.path.join(run_dir,"history.csv")
    if os.path.isfile(result_path) and not force_rerun:
        print(f"[resume] complete {cfg.name} seed {seed}")
        with open(result_path,"r",encoding="utf-8") as f:return json.load(f)
    set_seed(seed); model=E3PerceptionGNN(cfg,train_ds.global_mean_degree).to(device); params=parameter_count(model)
    preflight=equivariance_self_test(model,val_cache,cfg,vs,fs,device,batch_size)
    opt=torch.optim.Adam(model.parameters(),lr=protocol["learning_rate"],weight_decay=protocol["weight_decay"])
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=protocol["lr_factor"],patience=protocol["lr_patience_evals"],min_lr=protocol["min_lr"])
    criterion=nn.MSELoss(); best=float("inf"); best_epoch=0; no_imp=0; start_epoch=1; history=[]
    if resume and os.path.isfile(last_path) and not force_rerun:
        p=torch.load(last_path,map_location=device,weights_only=False)
        if p.get("config")==asdict(cfg) and int(p.get("seed"))==seed:
            model.load_state_dict(p["model_state_dict"]); opt.load_state_dict(p["optimizer_state_dict"]); sched.load_state_dict(p["scheduler_state_dict"]); best=float(p["best_val_mse"]); best_epoch=int(p["best_epoch"]); no_imp=int(p["evals_without_improvement"]); start_epoch=int(p["epoch"])+1; history=list(p.get("history",[])); print(f"[resume] {cfg.name} seed {seed} epoch {start_epoch}")
    print("\n"+"="*96); print(f"{cfg.name} | {cfg.family} | seed {seed} | params {params:,}"); print(json.dumps(asdict(cfg),indent=2)); print(f"mean edges {train_ds.mean_edges:.2f} | mean degree {train_ds.global_mean_degree:.3f} | TP {model.tp_weight_numel_per_layer}")
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    synchronize(); ttrain=time.perf_counter()
    for epoch in range(start_epoch,protocol["max_epochs"]+1):
        te=time.perf_counter(); model.train(); ss=0.; n=0; loader=make_loader(train_ds,True,seed+epoch,batch_size)
        for b in loader:
            b=b.to(device,non_blocking=True); opt.zero_grad(set_to_none=True); pred=model(b); loss=criterion(pred,b.y)
            if not torch.isfinite(loss): raise RuntimeError(f"Non-finite loss {cfg.name} seed {seed} epoch {epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),protocol["grad_clip_norm"]); opt.step(); e=pred.detach().float()-b.y.float(); ss += torch.sum(e**2).item(); n += e.numel()
        train_rmse=math.sqrt(ss/n); do_val=epoch==1 or epoch%protocol["val_every"]==0 or epoch==protocol["max_epochs"]; vm=vr=va=None; improved=False
        if do_val:
            vm,vr,va=evaluate_basic(model,make_loader(val_ds,False,0,batch_size),device); sched.step(vm)
            if vm < best-1e-15:
                best=vm; best_epoch=epoch; no_imp=0; improved=True; torch.save(dict(model_state_dict=model.state_dict(),config=asdict(cfg),seed=seed,epoch=epoch,val_mse=vm,parameter_count=params),best_path)
            else:no_imp+=1
        history.append(dict(epoch=epoch,train_rmse=float(train_rmse),val_rmse=None if vr is None else float(vr),val_mae=None if va is None else float(va),lr=float(opt.param_groups[0]["lr"]),improved=bool(improved),epoch_seconds=float(time.perf_counter()-te)))
        if do_val:
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),scheduler_state_dict=sched.state_dict(),config=asdict(cfg),seed=seed,epoch=epoch,best_val_mse=best,best_epoch=best_epoch,evals_without_improvement=no_imp,history=history),last_path); save_history_csv(history,hist_path)
            print(f"epoch {epoch:03d} | train {train_rmse:.6e} | val {vr:.6e} | MAE {va:.6e} | lr {opt.param_groups[0]['lr']:.2e} | {time.perf_counter()-te:.1f}s")
        if do_val and no_imp>=protocol["early_stop_patience_evals"]: print(f"early stop {epoch}; best {best_epoch}"); break
    synchronize(); train_seconds=time.perf_counter()-ttrain; peak=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    if not os.path.isfile(best_path): raise RuntimeError("No best checkpoint")
    bp=torch.load(best_path,map_location=device,weights_only=True); model.load_state_dict(bp["model_state_dict"]); model.eval()
    pred,target,tr,z,cache_idx,eval_seconds=predict_dataset(model,val_ds,device,batch_size); metrics=detailed_metric_dict(pred,target,tr,z,val_names); ckdiff=abs(metrics["component_rmse"]-math.sqrt(best))
    if ckdiff>5e-6: raise RuntimeError(f"Checkpoint reproduction failed {cfg.name} seed {seed}: {ckdiff:.3e}")
    e=pred-target; mse_node=torch.mean(e**2,dim=-1).numpy().astype(np.float32); rmse_node=np.sqrt(mse_node).astype(np.float32); vec=torch.linalg.vector_norm(e,dim=-1).numpy().astype(np.float32)
    err_dir=os.path.join(output_dir,"per_state_errors",safe_name(cfg.name)); os.makedirs(err_dir,exist_ok=True); err_path=os.path.join(err_dir,f"seed_{seed}.npz")
    np.savez_compressed(err_path,component_mse=mse_node,component_rmse=rmse_node,vector_error=vec,cache_index=cache_idx.numpy().astype(np.int32))
    stress=None if skip_symmetry else symmetry_stress_test(model,val_cache,cfg,vs,fs,device,batch_size,symmetry_graphs,transforms)
    ngraphs=len(val_ds); compute=dict(train_seconds=float(train_seconds),peak_vram_bytes=peak,inference_seconds_full_val=float(eval_seconds),inference_ms_per_graph=1000*eval_seconds/ngraphs,graphs_per_second=ngraphs/max(eval_seconds,1e-12),mean_edges_per_graph=val_ds.mean_edges,p95_edges_per_graph=val_ds.p95_edges,global_mean_degree=train_ds.global_mean_degree,edge_messages_per_graph=float(cfg.depth*val_ds.mean_edges),tp_weight_values_per_graph_estimate=float(val_ds.mean_edges*sum(model.tp_weight_numel_per_layer)),tp_weight_numel_per_layer=model.tp_weight_numel_per_layer)
    result=dict(experiment_name=cfg.name,family=cfg.family,seed=seed,config=asdict(cfg),parameters=params,best_epoch=best_epoch,best_val_mse=best,checkpoint_reproduction_difference=ckdiff,validation=metrics,preflight_equivariance=preflight,symmetry_stress=stress,compute=compute,protocol=protocol,per_state_error_file=err_path,checkpoint=best_path,epoch_cap_warning=bool(best_epoch>=protocol["max_epochs"]-protocol["val_every"]))
    json_dump(result,result_path); save_history_csv(history,hist_path)
    if os.path.isfile(last_path): os.remove(last_path)
    del model,opt,sched
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result

# -----------------------------------------------------------------------------
# Cross-run exports and oracle analysis
# -----------------------------------------------------------------------------
def load_completed_results(output_dir):
    out=[]
    for p in sorted(glob.glob(os.path.join(output_dir,"runs","*","seed_*","result.json"))):
        with open(p,"r",encoding="utf-8") as f: out.append(json.load(f))
    return out


def write_aggregate_outputs(results,output_dir):
    if not results:return
    path=os.path.join(output_dir,"aggregate_results.csv"); fields=["experiment","family","seed","parameters","best_epoch","component_rmse_A","component_mae_A","vector_error_mean_A","trajectory_rmse_mean_A","trajectory_rmse_std_A","train_seconds","peak_vram_bytes","inference_ms_per_graph","graphs_per_second","mean_edges_per_graph","edge_messages_per_graph","tp_weight_values_per_graph_estimate","epoch_cap_warning"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in results:
            m,c=r["validation"],r["compute"]; w.writerow(dict(experiment=r["experiment_name"],family=r["family"],seed=r["seed"],parameters=r["parameters"],best_epoch=r["best_epoch"],component_rmse_A=m["component_rmse"],component_mae_A=m["component_mae"],vector_error_mean_A=m["vector_error_mean"],trajectory_rmse_mean_A=m["trajectory_component_rmse_mean"],trajectory_rmse_std_A=m["trajectory_component_rmse_std"],train_seconds=c["train_seconds"],peak_vram_bytes=c["peak_vram_bytes"],inference_ms_per_graph=c["inference_ms_per_graph"],graphs_per_second=c["graphs_per_second"],mean_edges_per_graph=c["mean_edges_per_graph"],edge_messages_per_graph=c["edge_messages_per_graph"],tp_weight_values_per_graph_estimate=c["tp_weight_values_per_graph_estimate"],epoch_cap_warning=r["epoch_cap_warning"]))
    path=os.path.join(output_dir,"trajectory_results.csv"); fields=["experiment","seed","trajectory","component_rmse_A","component_mae_A","vector_error_mean_A","vector_error_p95_A","n_atom_instances"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in results:
            for tr,m in r["validation"]["per_trajectory"].items(): w.writerow(dict(experiment=r["experiment_name"],seed=r["seed"],trajectory=tr,component_rmse_A=m["component_rmse"],component_mae_A=m["component_mae"],vector_error_mean_A=m["vector_error_mean"],vector_error_p95_A=m["vector_error_p95"],n_atom_instances=m["n_atom_instances"]))
    path=os.path.join(output_dir,"compute_costs.csv"); base=["experiment","seed","parameters"]
    compute_keys=list(results[0]["compute"].keys())
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=base+compute_keys); w.writeheader()
        for r in results:
            row=dict(experiment=r["experiment_name"],seed=r["seed"],parameters=r["parameters"]); row.update(r["compute"])
            for k,v in list(row.items()):
                if isinstance(v,(list,dict)): row[k]=json.dumps(v)
            w.writerow(row)
    path=os.path.join(output_dir,"equivariance_results.csv"); fields=["experiment","seed","transform","kind","determinant","target_component_rmse_A","equivariance_component_rmse_A","equivariance_component_mae_A","accuracy_degradation_factor","n_graphs"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in results:
            s=r.get("symmetry_stress")
            if not s: continue
            for x in s["transforms"]: w.writerow(dict(experiment=r["experiment_name"],seed=r["seed"],transform=x["name"],kind=x["kind"],determinant=x["determinant"],target_component_rmse_A=x["target_component_rmse"],equivariance_component_rmse_A=x["equivariance_component_rmse"],equivariance_component_mae_A=x["equivariance_component_mae"],accuracy_degradation_factor=x["accuracy_degradation_factor"],n_graphs=s["n_graphs"]))


def grouped_seed_summary(results):
    by=defaultdict(list)
    for r in results: by[r["experiment_name"]].append(r)
    out={}
    for name,rows in by.items():
        rm=np.array([r["validation"]["component_rmse"] for r in rows]); lat=np.array([r["compute"]["inference_ms_per_graph"] for r in rows]); pars=np.array([r["parameters"] for r in rows])
        out[name]=dict(family=rows[0]["family"],n_seeds=len(rows),seeds=sorted(int(r["seed"]) for r in rows),rmse_mean=float(rm.mean()),rmse_std=float(rm.std(ddof=1) if len(rm)>1 else 0),latency_ms_mean=float(lat.mean()),latency_ms_std=float(lat.std(ddof=1) if len(lat)>1 else 0),parameters_mean=float(pars.mean()))
    return out


def write_pareto(results,output_dir):
    s=grouped_seed_summary(results); names=sorted(s); rows=[]
    for n in names:
        a=s[n]; dominated=False; dom=""
        for m in names:
            if m==n:continue
            b=s[m]
            if b["rmse_mean"]<=a["rmse_mean"] and b["latency_ms_mean"]<=a["latency_ms_mean"] and (b["rmse_mean"]<a["rmse_mean"] or b["latency_ms_mean"]<a["latency_ms_mean"]): dominated=True; dom=m; break
        rows.append(dict(experiment=n,**a,pareto=not dominated,dominated_by_example=dom))
    if rows:
        with open(os.path.join(output_dir,"pareto_frontier.csv"),"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader()
            for x in rows:
                y=dict(x); y["seeds"]=json.dumps(y["seeds"]); w.writerow(y)
    return rows


def posthoc_oracle(results,output_dir,state_descriptor_path):
    by=defaultdict(list)
    for r in results:
        if os.path.isfile(r.get("per_state_error_file","")): by[r["experiment_name"]].append(r)
    if not by:return None
    names=sorted(by); cols=[]; costs=[]; nseeds=[]; fam=[]; expected=None
    for name in names:
        stack=[]
        for r in by[name]:
            with np.load(r["per_state_error_file"]) as d: a=d["component_mse"].astype(np.float64)
            if expected is None: expected=len(a)
            if len(a)!=expected: raise RuntimeError("Per-state ordering length mismatch")
            stack.append(a)
        cols.append(np.mean(np.stack(stack),axis=0)); costs.append(np.mean([r["compute"]["inference_ms_per_graph"] for r in by[name]])); nseeds.append(len(by[name])); fam.append(by[name][0]["family"])
    mse=np.stack(cols,axis=1); rmse=np.sqrt(mse); costs=np.asarray(costs); normcost=costs/max(np.min(costs),1e-12)
    matrix_path=os.path.join(output_dir,"state_error_matrix.npz"); np.savez_compressed(matrix_path,component_mse=mse.astype(np.float32),component_rmse=rmse.astype(np.float32),config_names=np.asarray(names,dtype="U128"),family=np.asarray(fam,dtype="U64"),n_completed_seeds=np.asarray(nseeds,np.int16),inference_ms_per_graph=costs.astype(np.float32),normalized_cost=normcost.astype(np.float32))
    global_rmse=np.sqrt(np.mean(mse,axis=0)); gi=int(np.argmin(global_rmse)); out=dict(state_error_matrix=matrix_path,state_descriptors=state_descriptor_path,n_states=mse.shape[0],n_configs=mse.shape[1],configs=names,completed_seed_counts={n:int(s) for n,s in zip(names,nseeds)},global_best=dict(config=names[gi],component_rmse_A=float(global_rmse[gi])),lambda_sweep=[])
    with np.load(state_descriptor_path) as d: species=d["Z"].astype(np.int64); modes=d["loading_mode_code"].astype(np.int64)
    for lam in ORACLE_COST_LAMBDAS:
        choice=np.argmin(rmse+lam*normcost[None,:],axis=1); chosen=mse[np.arange(len(mse)),choice]; orm=float(np.sqrt(np.mean(chosen))); counts=Counter(names[int(i)] for i in choice.tolist())
        byz={}; bym={}
        for z in np.unique(species): byz[str(int(z))]={k:int(v) for k,v in Counter(names[int(i)] for i in choice[species==z]).most_common()}
        for m in np.unique(modes): bym[str(int(m))]={k:int(v) for k,v in Counter(names[int(i)] for i in choice[modes==m]).most_common()}
        out["lambda_sweep"].append(dict(lambda_value=float(lam),oracle_component_rmse_A=orm,absolute_gain_vs_global_best_A=float(global_rmse[gi]-orm),relative_gain_vs_global_best_percent=float(100*(global_rmse[gi]-orm)/global_rmse[gi]),mean_normalized_cost=float(np.mean(normcost[choice])),selection_counts={k:int(v) for k,v in counts.most_common()},selection_by_species_Z=byz,selection_by_loading_mode_code=bym))
    json_dump(out,os.path.join(output_dir,"oracle_results.json")); return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    root = ROOT_OVERRIDE or discover_root()
    if root is None:
        raise FileNotFoundError(
            "Could not auto-locate processed_production. Set ROOT_OVERRIDE at the top of the script."
        )

    root = os.path.abspath(root)
    outdir = os.path.abspath(OUTPUT_DIR_OVERRIDE or default_output_dir())
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if RUN_MODE not in {"full", "screen"}:
        raise ValueError(f"RUN_MODE must be 'full' or 'screen', got {RUN_MODE!r}")
    if SEED_LIMIT is not None and not (1 <= int(SEED_LIMIT) <= len(FIXED_TRAIN_SEEDS)):
        raise ValueError(f"SEED_LIMIT must be None or 1..{len(FIXED_TRAIN_SEEDS)}")

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("=" * 100)
    print("E3 PERCEPTION ABLATION — PART I")
    print("=" * 100)
    print("root", root)
    print("output", outdir)
    print("device", device)
    print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    print(
        "Python", platform.python_version(),
        "torch", torch.__version__,
        "e3nn", e3nn.__version__,
        "PyG", importlib.import_module("torch_geometric").__version__,
    )
    print("fixed seeds", FIXED_TRAIN_SEEDS)
    print("run mode", RUN_MODE)
    print("resume", RESUME)
    print("source graph cutoff", SOURCE_GRAPH_CUTOFF, "A")

    split, train_runs, val_runs, test_runs = load_split_manifest(root)
    print(
        f"train {len(train_runs)} | val {len(val_runs)} | "
        f"TEST {len(test_runs)} manifest-only; never resolved"
    )

    train_records = get_sample_records(root, train_runs)
    val_records = get_sample_records(root, val_runs)
    print("samples", len(train_records), len(val_records))

    train_cache, ts = load_cache(train_records, True, "Train cache")
    val_cache, vsmeta = load_cache(val_records, False, "Val cache")

    if not set(vsmeta["seen_species"]).issubset(ts["seen_species"]):
        raise RuntimeError("Validation unseen species")

    vscale = float(ts["velocity_scale"])
    fscale = float(ts["force_scale"])
    strides = set(ts["frame_strides"]) | set(vsmeta["frame_strides"])
    if len(strides) > 1:
        raise RuntimeError(f"Inconsistent strides {strides}")
    stride = next(iter(strides)) if strides else EXPECTED_FRAME_STRIDE
    dt = stride * MD_TIMESTEP_PS

    print(
        "TRAIN-only uncentered isotropic RMS",
        vscale, fscale,
        "means diagnostic only",
        ts["velocity_mean_diagnostic"],
        ts["force_mean_diagnostic"],
        "dt ps", dt,
    )

    observed = max(ts["max_stored_edge_distance"], vsmeta["max_stored_edge_distance"])
    print("max stored edge", observed, "declared source cutoff", SOURCE_GRAPH_CUTOFF)
    if SOURCE_GRAPH_CUTOFF < BASELINE_CUTOFF - 1e-8:
        raise RuntimeError("SOURCE_GRAPH_CUTOFF below baseline 3.5 A")

    # Guard against claiming a 5 A source graph while still using old 3.5 A data.
    inferred_round = math.ceil((observed - 1e-6) * 2) / 2
    if SOURCE_GRAPH_CUTOFF > inferred_round + 0.25:
        raise RuntimeError(
            f"SOURCE_GRAPH_CUTOFF={SOURCE_GRAPH_CUTOFF} A is inconsistent with "
            f"max stored edge {observed:.3f} A (inferred ~{inferred_round:.1f} A). "
            "Do not run larger-cutoff ablations on missing edges."
        )

    baseline_train = GraphViewDataset(train_cache, BASELINE_CUTOFF, vscale, fscale)
    bp = model_param_count_for_config(BASELINE_CONFIG, baseline_train.global_mean_degree)
    if bp != EXPECTED_BASELINE_C_PARAMS:
        raise RuntimeError(
            f"Baseline architecture regression: expected {EXPECTED_BASELINE_C_PARAMS}, got {bp}"
        )
    print("baseline architecture regression passed:", bp, "params")

    manifest = add_parameter_matched_controls(
        initial_manifest(SOURCE_GRAPH_CUTOFF),
        baseline_train.global_mean_degree,
    )
    families = (
        None
        if FAMILIES_TO_RUN == "all"
        else {x.strip() for x in FAMILIES_TO_RUN.split(",") if x.strip()}
    )
    configs = (
        None
        if not CONFIGS_TO_RUN
        else {x.strip() for x in CONFIGS_TO_RUN.split(",") if x.strip()}
    )
    selected = [
        c for c in manifest
        if (families is None or c.family in families)
        and (configs is None or c.name in configs)
    ]

    json_dump(
        dict(
            source_graph_cutoff_A=SOURCE_GRAPH_CUTOFF,
            all_configs=[asdict(c) for c in manifest],
            selected_configs=[c.name for c in selected],
            excluded_until_phase_II=[
                "winner-dependent targeted l x cutoff x depth interactions",
                "B-state reruns of selected winners",
                "learned controller / routing-predictability model",
            ],
        ),
        os.path.join(outdir, "manifest.json"),
    )

    print("\nMANIFEST")
    for i, c in enumerate(selected, 1):
        print(
            f"{i:02d} {'DEFER' if c.deferred_reason else 'RUN  '} "
            f"{c.name:34s} family={c.family:18s} cutoff={c.cutoff:.1f} "
            f"l={c.edge_lmax} depth={c.depth} rbf={c.num_radial} "
            f"agg={c.aggregation} hidden={c.hidden_irreps}"
            + (f" -> {c.deferred_reason}" if c.deferred_reason else "")
        )

    if LIST_CONFIGS_ONLY or DRY_RUN:
        print("\nPARAM COUNTS")
        for c in selected:
            if c.deferred_reason:
                continue
            d = GraphViewDataset(train_cache[:32], c.cutoff, vscale, fscale)
            m = E3PerceptionGNN(c, d.global_mean_degree)
            print(
                f"{c.name:34s} {parameter_count(m):,} params "
                f"TP={m.tp_weight_numel_per_layer}"
            )
        return

    desc_path = os.path.join(outdir, "state_descriptors.npz")
    if not os.path.isfile(desc_path) or FORCE_RERUN:
        print(
            "state descriptors",
            build_state_descriptors(val_cache, BASELINE_CUTOFF, desc_path),
        )

    protocol = training_protocol(RUN_MODE)
    seeds = (
        FIXED_TRAIN_SEEDS[: int(SEED_LIMIT)]
        if SEED_LIMIT is not None
        else list(FIXED_TRAIN_SEEDS)
    )
    transforms = make_stress_transforms(SYMMETRY_SO3, RANDOM_IMPROPER)
    val_names = {i: x["run_id"] for i, x in enumerate(val_runs)}

    json_dump(
        dict(
            root=root,
            output=outdir,
            device=str(device),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            python=platform.python_version(),
            torch=torch.__version__,
            e3nn=e3nn.__version__,
            torch_geometric=importlib.import_module("torch_geometric").__version__,
            mode=RUN_MODE,
            training_seeds=seeds,
            all_fixed_training_seeds=FIXED_TRAIN_SEEDS,
            protocol=protocol,
            train_trajectories=[x["run_id"] for x in train_runs],
            val_trajectories=[x["run_id"] for x in val_runs],
            test_trajectories_manifest_only=[x["run_id"] for x in test_runs],
            test_loaded=False,
            velocity_scale=vscale,
            force_scale=fscale,
            frame_stride=stride,
            delta_t_ps=dt,
            source_graph_cutoff_A=SOURCE_GRAPH_CUTOFF,
        ),
        os.path.join(outdir, "run_environment.json"),
    )

    t0 = time.perf_counter()
    resume = RESUME

    for ci, cfg in enumerate(selected, 1):
        if cfg.deferred_reason:
            print("[DEFERRED]", cfg.name, cfg.deferred_reason)
            continue

        train_ds = GraphViewDataset(train_cache, cfg.cutoff, vscale, fscale)
        val_ds = GraphViewDataset(val_cache, cfg.cutoff, vscale, fscale)
        print("\n" + "#" * 100)
        print(f"CONFIG {ci}/{len(selected)} {cfg.name}")
        print("#" * 100)

        for seed in seeds:
            train_one(
                cfg, seed, train_cache, val_cache, train_ds, val_ds,
                vscale, fscale, val_names, outdir, device, RUN_BATCH_SIZE,
                protocol, transforms, SYMMETRY_GRAPHS, SKIP_SYMMETRY,
                resume, FORCE_RERUN,
            )
            completed = load_completed_results(outdir)
            write_aggregate_outputs(completed, outdir)
            write_pareto(completed, outdir)

        completed = load_completed_results(outdir)
        posthoc_oracle(completed, outdir, desc_path)
        del train_ds, val_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    completed = load_completed_results(outdir)
    write_aggregate_outputs(completed, outdir)
    write_pareto(completed, outdir)
    oracle = posthoc_oracle(completed, outdir, desc_path)
    summary = grouped_seed_summary(completed)

    print("\n" + "=" * 100)
    print("PART-I COMPLETE FOR ALL CURRENTLY RUNNABLE CONFIGS")
    print("=" * 100)
    print(
        "records", len(completed),
        "minutes", (time.perf_counter() - t0) / 60,
        "TEST never resolved or loaded",
    )
    for name, stat in sorted(summary.items(), key=lambda kv: kv[1]["rmse_mean"]):
        print(
            f"{name:34s} n={stat['n_seeds']} "
            f"RMSE {stat['rmse_mean']:.8f} ± {stat['rmse_std']:.8f} A | "
            f"{stat['latency_ms_mean']:.4f} ms/graph"
        )
    if oracle:
        gb = oracle["global_best"]
        oz = oracle["lambda_sweep"][0]
        print("global best", gb)
        print(
            "statewise oracle lambda=0",
            oz["oracle_component_rmse_A"],
            "gain %",
            oz["relative_gain_vs_global_best_percent"],
        )


if __name__ == "__main__": main()
