# Governance and the privacy claim

## The artifact a hospital reads

Two objects, and neither is Python you have to follow line by line:

**The policy.** One `AperturePolicy`, declared where a reviewer can see it:
the byte cap per round, the shapes that may never leave, the exact list of
metric names permitted, and the minimum cohort behind any statistic.

**The log.** One append-only JSON-lines file per site, one record per
emission:

```json
{"array_count":1,"byte_count":57,"example_count":40,"kind":"fit",
 "metric_keys":["train_loss"],"payload_sha256":"6f0c…","round":1,
 "site_id":"north-general","status":"emitted",
 "timestamp":"2026-01-01T00:00:00+00:00"}
```

Rejections are written with `"status":"rejected"` and the reason. A log that
records only what succeeded answers half the question a reviewer is asking.

The payload digest covers the dtype, shape and bytes of each array plus the
metrics and the cohort size, so an aggregator's copy of a round can be checked
against the site's own record of what it sent.

## The privacy claim needs qualifying

Weights are not data, but they are not nothing. A generative model trained on
patient anatomy can in principle reproduce an identifiable individual, and
gradient-inversion attacks on federated learning are a live research area. A
public demonstration makes this a reputational question, not only an ethical
one.

Plan for it from Stage 2, not Stage 5:

- membership-inference evaluation as a standing test, not a one-off
- DP-SGD as an option with the utility cost measured and published
- secure aggregation as a stated integration point
- honesty in the write-up: *"no raw data left the site"* is true and provable
  from the audit log; *"the model cannot leak anything"* is neither

The aperture is a necessary boundary, not a sufficient one. It bounds and
records what leaves. It does not bound what can be inferred from what leaves.
Saying so plainly is part of the deliverable.

## Open questions

These are worth answering with two real sites before Stage 4, not after.

1. **Where does the cohort spec live?** A TOML file alongside the structure
   mapping, hashed into the manifest, seems right — but it overlaps with
   PyMedPhys's existing `config.toml` conventions.
2. **How much of the DICOM retrieve should PyMedPhys own?** A thin wrapper
   over `pynetdicom`, or a documented "point us at your local SCP"?
3. **Should evaluation be federated too?** A held-out set at each site gives
   honest per-site numbers, but multiplies the egress surface. The current
   answer is yes, and evaluation goes through the same aperture as training
   for exactly that reason.
4. **Cross-vendor.** Nothing here is Mosaiq-specific above the dataset layer —
   the structure canonicalisation deliberately lives under
   `pymedphys._dicom.structure`, not under `_mosaiq`. Is an ARIA cohort
   dataset in scope for someone else to contribute?
5. **What is the minimum viable governance artifact** a hospital would accept?

## Prior art worth reading before starting

- Flower, NVFlare, OpenFL — client/server federated learning frameworks;
  NVFlare has the most healthcare deployment experience.
- MONAI-FL and the federated segmentation consortium papers — closest existing
  work, mostly segmentation rather than generative.
- FedBN, FedProx — the two specific mitigations named in the design.
- TG-263 — the nomenclature standard the canonicalisation layer targets.
- Gradient-inversion and membership-inference literature — for the
  qualifications above, and to avoid overclaiming in the write-up.
