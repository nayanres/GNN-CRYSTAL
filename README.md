# Adaptive Equivariant Dynamics

**Learning transferable nonequilibrium atomistic dynamics with state-aware equivariant graph neural networks**

This project investigates whether graph neural networks can learn short-horizon nonequilibrium atomistic dynamics directly from molecular-dynamics trajectories and eventually generalize across materials, loading conditions, and material-material interactions.

The current work begins with a controlled crystalline-material benchmark and asks three progressively deeper questions:

1. **What dynamical state information is required to predict atomic evolution?**
2. **How much do interatomic interactions and geometric symmetry matter?**
3. **Should an equivariant model use the same geometric representation for every atomic environment, or should geometric computation adapt to the state being processed?**

The long-term goal is an extensible framework in which raw molecular-dynamics trajectories can be converted into standardized graph-state transitions and used to train transferable atomistic dynamics models.

---

## Core idea

At each timestep, an atomistic system is represented as a graph state:

$$
S_t =
\left(
Z,
\mathbf{r}_t,
\mathbf{v}_t,
\mathbf{F}_t,
H_t,
\text{global conditions}
\right)
$$

where:

- $Z$ denotes atomic species,
- $\mathbf{r}_t$ denotes atomic positions,
- $\mathbf{v}_t$ denotes velocities,
- $\mathbf{F}_t$ denotes forces,
- $H_t$ denotes the simulation cell,
- and global variables can encode loading, temperature, orientation, or other experimental conditions.

The model learns a transition operator:

$$
S_t \rightarrow \Delta \mathbf{r}_t
$$

or, in future versions:

$$
S_t \rightarrow S_{t+\Delta t}
$$

Rather than fitting an energy surface and subsequently integrating molecular dynamics, this project studies **direct prediction of dynamical evolution from trajectory data**.

---

# Current prototype

The current benchmark uses crystalline **anorthite, CaAl<sub>2</sub>Si<sub>2</sub>O<sub>8</sub>**, simulated with LAMMPS under controlled deformation.

## Dataset

The first production dataset contains:

- **36 independent MD trajectories**
- **18,000 graph-state transitions**
- **104 atoms per graph**
- **1,872,000 atom-level transition targets**
- 300 K thermal initialization
- isotropic and axis-specific deformation
- final deformation magnitudes of 0.5%, 1.0%, and 1.5%
- 5 ps loading trajectories
- graph states sampled every 0.01 ps

The data split is trajectory-level rather than transition-level:

```text
TRAIN: MD seed 11001
VAL:   MD seed 22002
TEST:  MD seed 33003
```

Adjacent MD frames are therefore never randomly divided between training and validation.

The test trajectories remain **sealed** while architecture and training decisions are still being developed.

This dataset currently measures:

> **Generalization to unseen thermal realizations under known material and loading conditions.**

It does **not** yet establish generalization to unseen materials, temperatures, loading modes, strain rates, or impact conditions.

---

# Data pipeline

The project includes an end-to-end pipeline:

```text
LAMMPS trajectory
        ↓
trajectory validation
        ↓
coordinate / cell normalization
        ↓
periodic graph construction
        ↓
state-transition dataset
        ↓
GNN training
        ↓
deterministic evaluation
        ↓
symmetry / trajectory / state analysis
```

The converter supports:

- orthogonal and triclinic simulation cells
- wrapped and unwrapped LAMMPS coordinates
- periodic neighbor graphs
- positions, velocities, forces, species, and cell information
- total, affine, and non-affine displacement decomposition
- trajectory-level metadata
- thermo data keyed by timestep
- strict finite-value and topology validation

The intended design principle is:

> **Nothing downstream of raw simulation generation should need to know that the material is anorthite.**

The same processing infrastructure should eventually support arbitrary crystalline materials and loading protocols.

---

# Preliminary findings

## 1. Dynamical state matters

Controlled state ablations compared three representations:

```text
A: species
B: species + velocity
C: species + velocity + force
```

Using the initial equivariant model:

| State | Validation RMSE |
|---|---:|
| Zero-displacement baseline | 0.03406 Å |
| A — species / geometry | 0.02983 Å |
| B — + velocity | 0.00388 Å |
| C — + force | 0.00202 Å |

Species and static geometry alone provide relatively little predictive information at this short horizon.

Velocity produces a very large improvement, and instantaneous force provides additional information.

This supports the interpretation:

$$
\boxed{\text{State representation is critical for direct dynamics prediction}}
$$

---

## 2. The learned model exceeds simple physical extrapolation

Physics-based short-horizon baselines were evaluated:

| Method | Validation RMSE |
|---|---:|
| Zero displacement | 0.03406 Å |
| Ballistic $v\Delta t$ | 0.01967 Å |
| Taylor / Verlet $v\Delta t + \frac{1}{2}a\Delta t^2$ | 0.01160 Å |
| Node-only equivariant model, state C | 0.00713 Å |
| Graph equivariant model, state C | 0.00202 Å |

The graph model substantially outperforms direct Taylor/Verlet extrapolation.

This suggests the model is learning useful interaction-dependent information beyond a trivial local integrator.

---

## 3. Interactions matter

A node-only equivariant model using species, velocity, and force reached approximately:

$$
0.00713\ \text{Å RMSE}
$$

while the graph model using the same state information reached approximately:

$$
0.00202\ \text{Å RMSE}
$$

The large difference indicates that neighbor interactions contain substantial predictive information even over a short timestep.

---

## 4. Symmetry matters

A matched non-equivariant NNConv model produced excellent canonical-orientation accuracy:

$$
\mathrm{RMSE} \approx 0.00132\ \text{Å}
$$

However, rotating the same physical systems produced strong prediction inconsistency.

Its mean SO(3) equivariance error was approximately:

$$
0.00303\ \text{Å}
$$

with approximately:

$$
2.38\times
$$

average degradation under rotations.

The E(3)/O(3)-equivariant model instead maintained numerical equivariance at approximately:

$$
10^{-7}\text{--}10^{-8}\ \text{Å}
$$

This originally exposed an important tradeoff:

```text
NNConv:
    better canonical-orientation accuracy
    worse rotational consistency

E3 equivariant model:
    slightly worse canonical accuracy
    essentially exact rotational / reflection consistency
```

---

# Updated equivariant baseline

The original equivariant experiments used a 200-epoch training budget.

A longer training study showed that this significantly underestimated the attainable accuracy of the equivariant architecture.

The current baseline configuration is:

```text
State:            species + velocity + force
Hidden irreps:    32x0e + 16x1o
Edge lmax:        2
Radial basis:     8
Message depth:    2
Cutoff:           3.5 Å
Aggregation:      mean
Parameters:       336,480
```

Within a fixed 400-epoch training budget, the best validation result reached:

$$
\boxed{\mathrm{RMSE}=0.001482\ \text{Å}}
$$

with:

$$
\mathrm{MAE}=0.001120\ \text{Å}
$$

Importantly, the best result occurred at the **400-epoch budget boundary**, so this should not yet be interpreted as a fully converged optimum.

This substantially reduces the previously observed canonical-accuracy gap between the equivariant model and NNConv.

It also demonstrates that training protocol can strongly confound architectural comparisons.

---

# Main current research question

The project is now focused on the following question:

> **Should equivariant graph neural networks use one globally fixed geometric representation, or should geometric resolution be allocated conditionally according to the atomic environment being processed?**

Standard equivariant atomistic architectures typically choose quantities such as:

- cutoff radius
- angular resolution
- hidden irreducible representations
- radial basis resolution
- message-passing depth
- aggregation method

globally for the entire model.

Every atom therefore receives approximately the same geometric processing regardless of whether its local environment is simple, distorted, rapidly evolving, highly anisotropic, or strongly interacting.

This project is investigating whether that assumption is unnecessarily restrictive.

---

# Part I — Equivariant graph-perception study

Before introducing an adaptive architecture, the project first measures which geometric design choices actually affect prediction.

The current study varies one component at a time while holding the dynamical state fixed.

## Aggregation

Planned comparisons include:

```text
mean
sum
global sqrt-degree normalization
local sqrt-degree normalization
learned invariant edge weighting
```

A key question is whether mean aggregation removes useful coordination information.

---

## Angular observation

The edge representation uses spherical harmonics:

$$
Y^{(l)}\left(\hat{\mathbf{r}}_{ij}\right)
$$

Experiments vary the angular bandwidth available to the network.

Planned comparisons include:

```text
edge lmax = 0
edge lmax = 1
edge lmax = 2
higher angular bandwidth configurations
```

---

## Hidden angular memory

The current hidden representation is:

$$
32\times 0e + 16\times 1o
$$

Future configurations retain higher-order representations inside the hidden state, including $l=2$ and $l=3$ channels.

For example:

$$
32\times 0e + 16\times 1o + 8\times 2e
$$

This tests whether the model currently observes angular information that it cannot preserve across layers.

Parameter-matched low-angular-order controls are included so improvements are not automatically attributed to increased parameter count.

---

## Radial perception

Experiments vary both cutoff radius and radial basis resolution.

### Cutoff radius

```text
2.5 Å
3.0 Å
3.5 Å
4.0 Å
4.5 Å
5.0 Å
```

The existing processed dataset contains edges through 3.5 Å.

Larger cutoffs will use a newly generated superset graph rather than reconstructing missing neighbors from the current graph files.

### Radial basis resolution

```text
4
8
16
32 radial basis functions
```

---

## Message-passing depth

Planned depths include:

```text
1
2
3
4
5 layers
```

This measures the spatial interaction range produced by repeated graph propagation separately from the explicit cutoff radius.

---

## Capacity

Width controls approximately spanning:

```text
0.5×
1×
2×
```

are included to distinguish genuine geometric effects from simple increases in model capacity.

---

# Per-state analysis

Global RMSE alone cannot answer whether adaptive computation is useful.

For every validation atomic state $i$ and architecture configuration $k$, the study records:

$$
e_{ik}
$$

States are aligned across models and include descriptors such as:

- atomic species
- $|\mathbf{v}|$
- $|\mathbf{F}|$
- $\mathbf{v}\cdot\mathbf{F}$
- coordination number
- neighbor-distance statistics
- local density
- loading direction
- deformation magnitude
- loading progress
- target displacement magnitude

This allows direct comparison of which environments benefit from which geometric representations.

For two configurations $A$ and $B$:

$$
\Delta_i = e_i^A - e_i^B
$$

The central empirical question is therefore not merely:

> Which architecture has the lowest average error?

but:

> **Do different physical states prefer different geometric computation?**

If one configuration dominates almost every state, an adaptive architecture may be unnecessary.

If different states systematically prefer different geometric resolutions, that provides direct evidence for adaptive graph perception.

---

# Part II — Adaptive Equivariant Graph Perception

If the perception study demonstrates state-dependent geometric requirements, the next model will introduce a learned invariant controller.

Conceptually:

$$
G_t
\xrightarrow{C_\phi}
\alpha_t
\rightarrow
F_\theta(G_t;\alpha_t)
\rightarrow
\Delta \mathbf{r}_t
$$

Here:

- $C_\phi$ is a state-dependent **invariant controller**
- $F_\theta$ is a shared **O(3)-equivariant dynamics network**
- $\alpha_t$ controls how much geometric computation is allocated

This is deliberately different from using independent mixture-of-experts networks.

All states retain access to the same shared learned representation.

The controller changes the **effective computation**, rather than selecting an isolated expert with a separate knowledge base.

---

## Adaptive angular resolution

One possible mechanism is dynamic weighting of angular channels:

$$
m_{ij}
=
\sum_l
\alpha_i^{(l)}
m_{ij}^{(l)}
$$

Simple environments may rely primarily on low-order information, while strongly anisotropic or distorted environments may activate higher angular orders.

---

## Adaptive radial perception

A maximum-radius graph can be constructed once.

An invariant controller then applies smooth edge gates:

$$
m_i
=
\sum_j
g_{ij}m_{ij}
$$

This allows the effective interaction radius to change continuously without dynamically rebuilding the graph.

---

## Adaptive depth

Residual message-passing blocks can be gated:

$$
h_i^{k+1}
=
h_i^k
+
\gamma_i^k F_k(h^k)
$$

Some environments may require deeper interaction propagation while others can be processed with fewer layers.

---

## Adaptive message allocation

Individual edges or interaction channels can receive learned invariant weights:

$$
\eta_{ij}
$$

This allows the network to concentrate computation on locally important interactions.

---

# Preserving equivariance

The controller must not break the physical symmetry of the base network.

For an orthogonal transformation $Q$, the controller should satisfy:

$$
C_\phi(QG)=C_\phi(G)
$$

while each geometric operator remains equivariant:

$$
F(QG)=QF(G)
$$

Invariant scalar gating then preserves equivariance:

$$
\alpha(QG)F(QG)
=
Q\left[\alpha(G)F(G)\right]
$$

Every candidate adaptive architecture will therefore be subjected to explicit numerical:

- SO(3) rotation tests
- inversion tests
- random improper O(3) transformation tests

rather than assuming equivariance solely from implementation.

---

# Adaptive computation objective

The eventual controller can be optimized using both prediction quality and computational cost:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{prediction}}
+
\lambda C
$$

The compute term $C$ can penalize expensive choices such as:

- higher angular momentum channels
- larger effective interaction radii
- more active message-passing blocks
- additional edges
- larger hidden representations

The intended objective is:

> **Learn the minimum sufficient geometric computation for each atomic environment.**

---

# Oracle analysis before controller training

Before training a controller, the fixed architecture experiments can define a statewise oracle:

$$
k_i^*(\lambda)
=
\underset{k}{\arg\min}
\left[
e_{ik}+\lambda C_k
\right]
$$

This provides an upper-bound diagnostic for adaptive computation.

If the oracle offers little benefit over the globally best model, adaptation is unlikely to justify its complexity.

If the oracle shows a substantial accuracy-compute advantage, there is measurable opportunity for a learned controller.

---

# Training-control experiments

The project is also investigating whether expensive architecture searches can be trained more efficiently.

Validation trajectories have shown repeated temporary regressions followed by later recovery.

For example, a training run can appear to plateau or degrade for several epochs and subsequently reach a substantially better minimum.

A future experiment will compare ordinary patience-based early stopping against a **recovery-confirmed stopping rule**:

```text
candidate plateau
      ↓
bounded forward recovery window
      ↓
meaningful recovery?
   /             \
 yes              no
  ↓                ↓
continue      confirm stop
```

A meaningful recovery will require an improvement larger than a predefined tolerance rather than floating-point noise.

This method will be evaluated against both:

- ordinary early stopping
- ordinary early stopping with a larger patience value

using:

- final validation RMSE
- GPU-hours
- premature-stop frequency
- number of recovered runs

This remains an experimental optimization method rather than an established contribution.

---

# Planned generalization hierarchy

The current benchmark represents only the first level of generalization.

Future experiments will progressively increase difficulty.

## G1 — unseen thermal trajectory

Current benchmark.

Train and validation trajectories use different thermal initializations while material and loading families remain known.

## G2 — unseen deformation magnitude

Hold out one or more strain magnitudes during training.

## G3 — unseen loading direction or mode

Train on a subset of loading modes and evaluate on unseen deformation directions.

## G4 — unseen thermodynamic or loading conditions

Examples include:

- temperature
- loading rate
- strain history
- pressure
- orientation

## G5 — unseen material

Train on multiple crystalline materials and evaluate on structures or compositions excluded from training.

## G6 — unseen material families

Evaluate extrapolation to compositions and structural families that differ substantially from the training distribution.

---

# Multi-material expansion

The next major dataset stage will include multiple materials and physical conditions.

Each sample will maintain a consistent state schema so the model does not depend on material-specific code.

A future dataset may include:

$$
\left\{
Z,
\mathbf{r},
\mathbf{v},
\mathbf{F},
H,
T,
\text{loading history},
\text{material metadata}
\right\}
$$

The architecture will then be evaluated for both interpolation and deliberate out-of-distribution transfer.

---

# Material-material interactions

A longer-term objective is modeling interactions between separate material bodies.

The system state can be represented as:

$$
S =
S_A
+
S_B
+
\text{relative interaction conditions}
$$

Potential global features include:

- relative velocity
- interaction normal
- orientation
- body identity
- temperature
- loading history

A particularly strong transfer experiment would train on combinations such as:

```text
A-A
A-C
B-B
B-C
```

and evaluate on an unseen interaction:

```text
A-B
```

This tests compositional generalization rather than memorization of known pairings.

---

# Rollout experiments

Current results primarily measure one-step prediction.

Future autonomous or semi-autonomous models must also be tested through repeated rollout.

Metrics will include:

- error versus rollout horizon
- rollout stability
- displacement distributions
- radial distribution functions
- coordination-number evolution
- structural statistics
- thermodynamic quantities where applicable
- failure or divergence rate

Long-horizon atom-by-atom correspondence alone is not sufficient because atomistic dynamics can be chaotic.

The objective is therefore both trajectory fidelity and preservation of physically meaningful statistical behavior.

---

# Important force-input limitation

The strongest current state representation includes the true instantaneous MD force:

$$
(Z,\mathbf{v},\mathbf{F})
$$

This makes the current state-C architecture useful as:

- a state-sufficiency experiment
- a learned short-step propagator
- a hybrid learned integrator

but it is not yet a fully autonomous force-free simulator.

Future work will therefore also study:

## State B

$$
(Z,\mathbf{v})
$$

which removes explicit force input.

## Force prediction

Predicting forces or accelerations jointly with state evolution.

## Hybrid simulation

Coupling a learned transition model with an external force calculator.

---

# Fresh out-of-distribution evaluation

The current validation set is intentionally used for model development.

The existing test split remains untouched during architecture design.

For stronger final evaluation, the project also plans to generate new molecular-dynamics trajectories **after model and training decisions have been frozen**.

A possible protocol is:

```text
1. finalize architecture
2. finalize preprocessing
3. finalize training protocol
4. freeze model code
5. record checkpoint / source hash
6. generate fresh OOD MD trajectories
7. evaluate once
```

This reduces the risk of repeatedly tuning against a static benchmark.

---

# Reproducibility goals

The project aims to maintain:

- explicit train/validation/test trajectory manifests
- deterministic dataset conversion
- fixed model-training seeds
- train-only normalization
- exact architecture configuration records
- checkpoint reproduction tests
- numerical O(3) symmetry tests
- per-trajectory metrics
- aligned per-state error files
- runtime and memory measurements
- dataset provenance
- software and environment metadata

Current fixed model-training seeds are:

```text
9078
4577
3320
3733
9428
```

These are distinct from molecular-dynamics trajectory seeds.

---

# Near-term roadmap

## Phase 1 — Controlled single-material study

- [x] LAMMPS trajectory generation
- [x] general trajectory-to-graph converter
- [x] trajectory-level train/validation/test split
- [x] physics baselines
- [x] state A/B/C ablation
- [x] node-only interaction control
- [x] equivariant vs non-equivariant symmetry comparison
- [x] extended equivariant baseline training
- [ ] complete one-seed geometric-perception screening
- [ ] construct 5 Å superset graphs
- [ ] analyze aligned per-state architecture errors
- [ ] identify geometric settings with state-dependent advantages
- [ ] replicate promising configurations across multiple seeds

## Phase 2 — Adaptive equivariant architecture

- [ ] implement invariant controller
- [ ] begin with adaptive angular / irrep allocation
- [ ] verify exact numerical O(3) behavior
- [ ] compare against globally fixed architecture
- [ ] introduce computation regularization
- [ ] evaluate adaptive radial perception
- [ ] evaluate adaptive depth
- [ ] evaluate adaptive message weighting
- [ ] compare learned controller against statewise oracle

## Phase 3 — Broader physical generalization

- [ ] add additional crystalline materials
- [ ] vary temperature
- [ ] vary loading rates
- [ ] hold out deformation modes
- [ ] hold out materials
- [ ] test unseen structural/compositional families
- [ ] run multi-step rollout studies

## Phase 4 — Interaction and impact systems

- [ ] construct multi-body atomistic states
- [ ] generate material-material interaction trajectories
- [ ] test held-out material pairings
- [ ] study extreme transient loading
- [ ] evaluate transfer across interaction geometry and conditions

---

# Long-term objective

The eventual goal is a transferable atomistic dynamics framework capable of learning from standardized nonequilibrium trajectory datasets across many materials and physical regimes.

Conceptually:

$$
\boxed{
\text{material state}
+
\text{interaction conditions}
\rightarrow
\text{future atomistic response}
}
$$

Rather than building a separate model for every material or experiment, the project investigates whether a shared equivariant architecture can learn reusable dynamical structure across heterogeneous physical systems.

The central architectural hypothesis is:

$$
\boxed{
\text{Geometric computation should adapt to the local physical state}
}
$$

rather than remaining globally fixed for every atom, at every timestep, in every material.

---

# Potential applications

At sufficient scale and validation, the framework may be relevant to:

- computational materials discovery
- accelerated molecular-dynamics surrogates
- nonequilibrium material response
- high-strain-rate and transient phenomena
- interfaces and heterogeneous materials
- aerospace materials
- high-temperature structural systems
- multiscale simulation
- digital engineering
- rapid screening of material response

These are long-term targets rather than claims about the present single-material prototype.

---

# Current status

This repository should be considered an **active research prototype**.

The strongest conclusions currently supported are:

1. dynamical state information strongly affects short-horizon prediction;
2. interatomic graph interactions provide substantial information beyond node-local state;
3. exact geometric equivariance eliminates severe rotational inconsistency observed in a matched non-equivariant model;
4. training budget materially affects apparent architecture performance;
5. the remaining open question is whether different atomic environments require different levels of geometric representation and computation.

The next stage is therefore not simply increasing model size.

It is determining:

> **What geometric information is necessary, for which atomic states, and whether a model can learn to allocate that computation automatically?**
