# Mosaiq-backed DICOMweb service (scaffold)

This module exposes an Elekta Mosaiq oncology information system (OIS) over
the [DICOMweb](https://www.dicomstandard.org/using/dicomweb) RESTful
services. Mosaiq's relational schema is translated on the fly into the
DICOM JSON model (PS3.18 Annex F).

## Status

| Service   | Endpoint(s)                                  | Status                         |
| --------- | -------------------------------------------- | ------------------------------ |
| QIDO-RS   | `GET /studies`                               | **Implemented** (study search) |
| QIDO-RS   | `GET /studies/{uid}/series`, `.../instances` | Scaffold (returns no matches)  |
| WADO-RS   | `GET /studies/{uid}`                         | Scaffold (`501`)               |
| STOW-RS   | `POST /studies`                              | Scaffold (`501`)               |

QIDO-RS study search supports the `PatientID`, `PatientName` and
`StudyDate` matching keys (with DICOM `*`/`?` wildcards and date ranges),
plus the `limit` and `offset` control parameters.

## Layout

| File         | Responsibility                                             |
| ------------ | ---------------------------------------------------------- |
| `mapping.py` | Pure Mosaiq-row → `pydicom.Dataset` translation (testable) |
| `qido.py`    | QIDO-RS → parameterised Mosaiq SQL, row → DICOM JSON        |
| `wado.py`    | WADO-RS retrieve scaffold                                  |
| `stow.py`    | STOW-RS store scaffold                                     |
| `server.py`  | Flask application factory wiring the DICOMweb URI routes   |

## Running

```bash
pymedphys dicomweb serve <mosaiq-hostname> --database MOSAIQ --http-port 8008
```

Then query it like any DICOMweb origin server:

```bash
curl "http://127.0.0.1:8008/studies?PatientID=MR8002"
```

Programmatic use:

```python
import pymedphys.mosaiq
import pymedphys.dicomweb

connection = pymedphys.mosaiq.connect("mosaiq-hostname")
app = pymedphys.dicomweb.create_app(get_connection=lambda: connection)
app.run(port=8008)
```

## Extension points

The scaffold marks, in code, where real Mosaiq deployments plug in:

- **UIDs** — `mapping.UID_ROOT` / `mapping.derive_uid`. Study/series/instance
  UIDs are derived deterministically from Mosaiq primary keys. Replace with
  the real DICOM UIDs where a site stores them, and swap `UID_ROOT` for an
  organisation-specific root.
- **Study definition** — a study currently maps to a Mosaiq treatment `Site`.
  Adjust the join in `qido.search_for_studies` to match local conventions.
- **Series / instances** — `qido.search_for_series` /
  `search_for_instances` return no matches. Map `TxField` (or imaging
  series) to DICOM series here.
- **WADO-RS / STOW-RS** — `wado.py` / `stow.py`. Implement SOP-instance
  synthesis (e.g. RT Plan from Mosaiq) or proxy to a PACS.
