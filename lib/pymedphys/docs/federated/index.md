# A Federated Learning Tap

```{warning}
Draft and prototype. The public surface lives in `pymedphys.beta.federated`
and is expected to churn. Nothing here is a claim of regulatory clearance;
this is research infrastructure.
```

A clinic-side boundary that lets a model be trained across clinics without
patient data leaving any of them.

## Motivation

Every interesting model in radiotherapy is data-starved for the same reason:
the data exists, in quantity, inside clinical systems that will never export
it. Public head-and-neck segmentation sets are small, old, and encumbered.
Meanwhile a single Mosaiq instance may hold tens of thousands of planned
patients that no researcher can touch.

Federated learning inverts the transfer: instead of moving data to the model,
move the model to the data and move back only what was learned. PyMedPhys is
an unusually good home for the clinic-side half of this, because it already
lives where the data is — it speaks Mosaiq SQL, DICOM, and TG-263, and it is
already installed on physics workstations inside hospital networks.

What this adds is *not* an aggregation server and *not* a model zoo. It is a
small, auditable boundary — the **tap** — plus the extraction path that fills
it.

### The tap is an aperture, not a pipe

A `DataLoader` exists to widen a pipe: get more data to the model, faster. A
federated tap exists to *narrow* one. The engineering artifact of value is the
aperture — the precise, enforced, logged specification of which bytes are
permitted to leave a clinic. Everything else in this design is bookkeeping
around that one object.

This framing matters for review. A physicist or ethics committee asked to
approve this should not have to read a training loop. They should have to read
one policy object and one audit log.

## Scope and non-goals

**In scope**

- Cohort selection and data extraction from Mosaiq (SQL index + DICOM payload).
- Canonicalisation to a declared common representation (grid, spacing, TG-263).
- A framework-agnostic `ClinicTrainer` contract.
- An egress policy + audit mechanism (the aperture).
- Thin adapters to at least one federated learning framework (Flower first).
- A worked demo: a 3D VAE over CT + OARs trained across simulated clinics.

**Out of scope, deliberately**

- Running or hosting an aggregation server. Clinics federate with whoever they
  choose; PyMedPhys ships the client half.
- Model architectures as library API. The demo VAE is an example, not a
  supported model.
- Secure multi-party aggregation primitives. We define where they plug in and
  defer to specialist libraries.
- Any claim of regulatory clearance.

## Try it

Stage 0 runs on NumPy alone — no server, no framework, no data:

```bash
python -m pymedphys._federated.demo
```

Three simulated non-IID clinics, in-process FedAvg over eight rounds converging
on the pooled mean; the manifest gate rejecting a clinic with a different
structure vocabulary; the aperture rejecting both a voxel-shaped array and a
metric key that is not on the whitelist; and an audit line per emission.

One result from that run is carried forward into the demo rather than hidden:
per-clinic evaluation loss **diverges** across rounds while the global parameter
converges, and no clinic does as well as it would have alone. That is the honest
picture of federating heterogeneous clinics — the global model is a compromise
that none of the participants would have chosen for themselves. Showing it
builds more trust than smoothing it.

```{toctree}
:maxdepth: 2

design
governance
roadmap
```
