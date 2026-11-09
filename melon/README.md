# melon: template transfer workflow

Move a template run collection from the NERSC `decam` repo to `main` at USDF, via a
relocatable sqlite butler.

Example chained collection: `u/seanmacb/DECam/calib/ELAIS_pilot_w40`
(sqlite butler is named `ELAIS_pilot_w40`; run collection transferred is `<chain>/template`).

| Step | Where | Script |
|------|-------|--------|
| 1. create sqlite butler named after the chain | NERSC | `01_create_butler.sh` |
| 2. register DECam instrument | NERSC | `02_register_instrument.sh` |
| 3. `transfer-datasets` from `decam` with `-t copy` | NERSC | `03_transfer_from_nersc.sh` |
| 4. globus the butler directory to USDF | manual | n/a |
| 5. `transfer-datasets` from the sqlite butler into `main` | USDF | `05_transfer_to_main.sh` |
| 5b. or: ingest into `main` under a renamed run (`u/<other>/X` -> `u/$USER/X`) | USDF | `05b_ingest_renamed.py` |

`run_nersc_steps.sh <chain> [short-name]` runs 1-3 back to back.

```bash
./run_nersc_steps.sh u/seanmacb/DECam/calib/ELAIS_pilot_w40
# ... globus melon/pilot/ELAIS_pilot_w40 to USDF ...
DRY_RUN=1 ./05_transfer_to_main.sh u/seanmacb/DECam/calib/ELAIS_pilot_w40   # at USDF
./05_transfer_to_main.sh u/seanmacb/DECam/calib/ELAIS_pilot_w40
```

## Configuration

Each script takes the full chained collection and an optional short name. The short name
defaults to the last path component and cannot contain `/`. Other settings are env vars
(see the header of `common.sh`): `BUTLER_PARENT` (default `melon/pilot`), `SOURCE_REPO`
(`decam`), `COLLECTIONS` (`<chain>`; in step 5, `*`), `DATASET_TYPE` (`template_coadd`),
`WHERE` (`skymap='lsst_cells_v2'`), `MAIN_REPO` (`main`), `MAIN_TRANSFER` (`copy`),
`REPO_OVERRIDE` (step 5 only, path to the sqlite butler at USDF).

## Notes

- Requires the LSST stack (`butler` on `PATH`) at both sites.
- The datastore root is `<butlerRoot>`-relative and `copy` mode puts files under it, so the
  butler directory can be moved with globus as is.
- The butler directory holds the datasets themselves. Keep it out of git, and out of
  whatever syncs this scripts directory to USDF, since globus carries it.
- Step 5 does not create a chained collection in `main`. Only the run collection
  is transferred. Define the chain there separately if you need it.
- Step 3 copies only the RUN collections behind the chain, so the chain name does not exist in
  the sqlite butler. Step 5 therefore matches `COLLECTIONS` (names or globs, comma-separated)
  against the sqlite butler's RUN collections, defaulting to all of them. Watch out for a
  `COLLECTIONS` still exported in your shell from an earlier transfer.
- Files keep their `<run>/...` path in `main`'s datastore (`/sdf/data/rubin/repo/main_20210215`),
  so you need write access to the run's top directory there. `u/<you>/` is fine; another
  user's namespace (e.g. `u/descdm/`) needs a repo admin to create it for you. Step 5 checks
  this up front and stops with an error if it would fail.
- To avoid that, use `05b_ingest_renamed.py <sqlite butler path> [--dry-run]` instead of step 5.
  It ingests the files into `main` under `u/$USER/...` with new (deterministic) dataset IDs,
  hardlinking by default: no extra disk, and the sqlite butler stays valid and can be deleted
  later without affecting `main`. It ingests in batches and skips what is already there, so
  just rerun it after an interruption. The FITS headers keep the original run/ID provenance.
