# Design

```
┌─ inside the clinic ──────────────────────────────────────────┐
│                                                              │
│  Mosaiq SQL ──┐                                              │
│   (cohort)    ├──► MosaiqCohortDataset ──► SiteTrainer ──┐   │
│  Mosaiq DICOM ┘     (canonicalise)         (torch, etc)  │   │
│   (CT, RTSTRUCT)                                         │   │
│                                                      ┌───▼─┐ │
│                                                      │ Ap- │ │
│                                                      │ ert-│ │
│                                                      │ ure │ │
│                                                      └───┬─┘ │
└──────────────────────────────────────────────────────────┼───┘
                                                           │
                              weights + whitelisted metrics│
                                                           ▼
                                         Flower / NVFlare / OpenFL
                                              aggregator
```

Three properties fall out of this shape:

1. **The dataset object is the only thing that sees voxels.** It is also the
   only thing that needs clinic-specific configuration.
2. **The trainer knows nothing about the network.** It can be exercised end to
   end in a unit test with no server running.
3. **The aperture is the only egress.** Not a convention — the trainer's `fit`
   returns whatever `aperture.emit()` returns, so bypassing it is a visible
   code change, not an omission.

## Getting the data out of Mosaiq

This is the part most likely to surprise contributors, so it is stated first.

`pymedphys.mosaiq` today is a SQL connection plus a query executor. That gives
you the **index** — patient, course, site, machine, plan approval state,
treatment dates — and nothing else. CT pixel data and RTSTRUCT contours are
not in reach of a `SELECT`.

So extraction is two-legged:

| Leg | Source | Gives you |
| --- | --- | --- |
| Index | Mosaiq SQL via `pymedphys.mosaiq` | cohort definition, provenance, stratification keys |
| Payload | Mosaiq DICOM service (C-FIND/C-MOVE) or the image archive | CT series, RTSTRUCT |

The index leg is where cohorts get defined reproducibly: *"approved H&N plans,
2019–2024, this machine group, excluding re-treats."* The payload leg is a
DICOM Q/R against the Mosaiq DICOM node, retrieved to a local SCP.

The deliverable for this leg is `MosaiqCohortDataset`, which is independently
useful to PyMedPhys users who have no interest in federation at all. It should
be reviewable and mergeable on its own. It is Stage 1 of the roadmap and is
not yet implemented.

## Canonicalisation, and why nomenclature is the real problem

The literature frames non-IID federated learning as a statistical problem.
In radiotherapy it is overwhelmingly a *naming and geometry* problem:

- `Parotid_L` vs `L Parotid` vs `PAROTID LT` vs `Lt_Parotid_gland`
- 2 mm vs 3 mm slice thickness; 512² vs 1024² acquisition
- different scanners, kVp, and reconstruction kernels
- OARs contoured to different atlases and margins

If sites silently disagree on any of this, training does not crash. It
produces a worse model and a plausible loss curve, which is far more expensive
than a crash.

**Mitigation:** every site declares its representation in a `SiteManifest` —
grid shape, voxel spacing, sorted canonical structure keys, and the SHA-256 of
the structure vocabulary it maps onto. These hash to a single
`compatibility_key`. The aggregator compares keys before round 1. A mismatch
is a startup error naming the disagreeing sites *and the fields they disagree
on*, because "training failed" is not an actionable error at 2am in a
hospital.

### One mapping file, two hashes

The original proposal put the SHA-256 of "the TG-263 mapping file" into the
compatibility key. Implementing that literally does not work: every site's
mapping file legitimately differs — differing local names are the entire
reason the file exists — so a whole-file hash would never match across two
clinics and the gate would reject every real federation.

So the file has two parts, hashed separately:

`[vocabulary]`
: The shared target list, and its version. Every participating site holds the
  same one. Its hash **is** part of the compatibility key.

`[aliases]`
: The site's own local names for those targets. Its hash is recorded in the
  manifest and the audit trail for provenance, and is **not** compared between
  sites.

```toml
[vocabulary]
version = "2026-01"
structures = ["Brainstem", "Parotid_L", "Parotid_R", "SpinalCord"]

[aliases]
Parotid_L = ["Lt_Parotid_gland", "L PAROTID"]
SpinalCord = ["Cord"]
```

Both hashes are taken over a canonical serialisation rather than the file
bytes, so reformatting a TOML file does not eject a clinic from a federation.

Automatic normalisation
(`pymedphys._dicom.structure.tg263.normalise`) resolves case, punctuation,
whitespace and laterality wording — `Parotid_L`, `L Parotid`, `PAROTID LT` and
`parotid-left` all reach the same key — and deliberately stops there. Anything
beyond that is clinical knowledge. It belongs in the hand-maintained,
version-controlled alias table, and it should be reviewed by a human at each
clinic, once. That is not a weakness of the design; it is the design.

A local name that is neither canonical nor aliased raises `UnmappedStructure`.
Silently dropping a structure a clinician contoured is exactly the failure
this module exists to prevent.

## The aperture

Enforced on every outbound payload, training and evaluation alike:

- **Size cap.** Total bytes per round, hard.
- **Forbidden shapes.** Any array whose shape matches the data grid is
  rejected — including one whose *trailing* axes match, which catches a batch
  of volumes. Cheap, and catches the class of bug where a debugging tensor
  ends up in the return value.
- **Metric key whitelist.** Only declared keys travel. `patient_mrn` is not a
  metric. A whitelisted key may not smuggle a non-scalar either.
- **Minimum cohort for statistics.** A mean over n=1 is a data leak wearing a
  moustache.
- **Finiteness.** Off by default is the wrong default here: a NaN payload
  poisons an aggregate silently and the check is close to free.

`policy_from_manifest` derives the forbidden shapes from the site's own
declared grid, so the two cannot drift apart.

And recorded, append-only, per emission: timestamp, site, round number, kind,
status, array count, byte count, example count, metric keys, and the SHA-256
of the payload. Rejections are recorded too, with their reason — a log that
only shows what succeeded answers half the question.

The audit log is the deliverable that makes this defensible. It answers "what
left this hospital, when, and how big was it" without anyone reading Python.

## The `SiteTrainer` contract

Six methods. Framework-agnostic, NumPy in and out.

```python
class SiteTrainer(Protocol):
    def manifest(self) -> SiteManifest: ...
    def shared_keys(self) -> list[str]: ...      # what crosses the boundary
    def get_weights(self) -> list[np.ndarray]: ...
    def set_weights(self, weights) -> None: ...
    def fit(self, config) -> FitResult: ...
    def evaluate(self, config) -> EvalResult: ...
```

`shared_keys()` is doing more work than it looks. It is how you express:

- **FedBN** — keep normalisation statistics site-local by excluding them.
- **Partial federation** — share a decoder, keep an encoder private.
- **Staged unfreezing** — widen the shared set as trust or evidence grows.

It is also compared across sites before round 1: two clinics that disagree on
what is shared are not running the same experiment.

`config` flows from the server to every site each round. That is how a
schedule such as a KL warm-up stays in step — `beta` is decided centrally, not
per site.

## Dependency policy

PyMedPhys will not take `torch` as a hard dependency, and should not. Nor
`flwr`.

- `protocol.py`, `aperture.py`, `simulate.py`, `toy.py`, `demo.py` — NumPy
  only, via the lazy `pymedphys._imports` mechanism. Always importable.
- `torch_site.py`, `flower_adapter.py` — lazy imports behind an extra,
  `pymedphys[federated]`. Not yet written.
- The public surface is importable, documented and testable in an environment
  with neither installed. There is a standing test asserting exactly that.

## Why Flower first

Smallest adapter, pure Python, simulation backend built in, no orchestration
assumptions. NVFlare is the more natural fit for a real multi-hospital
deployment with healthcare IT involved, and the adapter for it should be
roughly the same size. Framework choice is deliberately a leaf, not a root.

## VAE-specific hazards

- **BatchNorm under FedAvg** averages running statistics across sites with
  different scanners. Use GroupNorm. If BatchNorm is unavoidable, exclude it
  from `shared_keys()` (FedBN).
- **Posterior collapse** is worse federated than centralised. Warm up the KL
  weight β over rounds, driven from the server's round config so all sites
  stay in step.
- **Client drift** on strongly non-IID anatomy. FedProx (`proximal_mu`) is the
  cheap first lever; consider it before more exotic aggregation.
- **Aggregating a latent space is not obviously meaningful.** Averaging
  encoder weights across sites whose posteriors have drifted may produce a
  latent geometry that is a compromise of two coherent spaces and itself
  coherent in neither. Sharing only the decoder is a defensible fallback, and
  `shared_keys()` is how you say so.
