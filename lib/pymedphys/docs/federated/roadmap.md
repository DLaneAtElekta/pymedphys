# Roadmap

Each stage is independently valuable and independently reviewable. Do not
build stage *n+1* before stage *n* runs.

## Stage 0 — plumbing, no model, no data ✅

`ClinicTrainer`, `ClinicManifest`, `Aperture`, in-process FedAvg, structure
canonicalisation, and a NumPy toy trainer that fits a mean vector. Proves the
contract, the manifest gate, and the aperture guards with zero heavyweight
dependencies. Runs in under a second.

**Exit criterion:** federated mean over three non-IID clinics matches the pooled
mean; both guards raise on cue; audit log is written. *Met — asserted in
`lib/pymedphys/tests/federated/`, and demonstrated by
`python -m pymedphys._federated.demo`.*

What landed:

| Module | Contents |
| --- | --- |
| `pymedphys._federated.protocol` | `ClinicTrainer`, `ClinicManifest`, `FitResult`, `EvalResult` |
| `pymedphys._federated.aperture` | `AperturePolicy`, `Aperture`, `ApertureViolation`, audit log |
| `pymedphys._federated.simulate` | in-process FedAvg, manifest compatibility gate |
| `pymedphys._federated.toy` | NumPy trainer for tests and teaching |
| `pymedphys._federated.demo` | the runnable Stage 0 demonstration |
| `pymedphys._dicom.structure.tg263` | structure canonicalisation + vocabulary hashing |
| `pymedphys.beta.federated` | the public, churn-expected surface |

## Stage 1 — `MosaiqCohortDataset`

SQL cohort query → DICOM retrieve → resample → canonicalise → cache to disk as
`(CT, structure masks, metadata)`. Test against `pymedphys dev mssql` plus a
local DICOM SCP (Orthanc). The mapping file format and its hashing are already
in place from Stage 0; what remains is the two-legged extraction.

**Exit criterion:** a reproducible cohort spec file yields a byte-identical
cached tensor set on re-run, and the manifest hash is stable.

## Stage 2 — centralised baseline

Train the VAE on one clinic's cached data, single process, no federation. This
is not a detour — federated numbers are meaningless without it, and the
reconstruction quality ceiling is set here.

**Exit criterion:** recorded loss curves and sample reconstructions that a
physicist would call anatomically plausible.

## Stage 3 — simulated federation

Split one physical dataset into N virtual clinics. Split by *treating machine*,
*era*, or *contouring clinician* — never randomly. Random splits produce IID
data and hide precisely the failure modes worth finding.

Run through the in-process loop from Stage 0 first, then Flower's simulation
backend. No networking, no IT ticket, full debugger.

**Exit criterion:** federated model within a stated margin of the centralised
baseline, and a demonstration that a deliberately mis-mapped clinic is rejected
before round 1.

## Stage 4 — two processes, then two hosts, then two clinics

Flower gRPC. Introduce TLS, retry/dropout handling, and per-round resource
limits. Governance review happens here, with the audit log as the artifact.

## Stage 5 — the demonstration

Sample the aggregate latent space and show the model generating anatomies that
belong to neither contributing clinic. That is the claim worth making: the model
learned something no single clinic could have taught it.
