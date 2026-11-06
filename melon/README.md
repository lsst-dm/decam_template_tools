# melon: template transfer workflow

Move a template run collection from the NERSC `decam` repo to `main` at USDF, via a
relocatable sqlite butler.

Example chained collection: `u/seanmacb/DECam/calib/ELAIS_pilot_w40`
(sqlite butler is named `ELAIS_pilot_w40`; the datasets are taken from the chain itself, or
from `<chain>/template` if you set `COLLECTIONS` to it).

| Step | Where | Script |
|------|-------|--------|
| 1. create sqlite butler named after the chain | NERSC | `01_create_butler.sh` |
| 2. register DECam instrument | NERSC | `02_register_instrument.sh` |
| 3. `transfer-datasets` from `decam` with `-t copy` | NERSC | `03_transfer_from_nersc.sh` |
| 4. globus the butler directory to USDF | manual | n/a |
| 5. `transfer-datasets` from the sqlite butler into `main` | USDF | `05_transfer_to_main.sh` |

`run_nersc_steps.sh <chain> [short-name]` runs 1-3 back to back. `run-nersc-job.sh` does the same
as a Slurm job (`MAIL_TO=<address> ./run-nersc-job.sh <chain> [short-name]`): it defaults
`SOURCE_REPO` to the CFS decam repo and `BUTLER_PARENT` to `$PSCRATCH/melon`, and mails
`MAIL_TO` only if the source and sqlite dataset counts match.

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
(`decam`), `COLLECTIONS` (the chain), `DATASET_TYPE` (`template_coadd`),
`WHERE` (`skymap='lsst_cells_v2'`), `MAIN_REPO` (`main`), `MAIN_TRANSFER` (`copy`),
`REPO_OVERRIDE` (step 5 only, path to the sqlite butler at USDF).

## Notes

- Requires the LSST stack (`butler` on `PATH`) at both sites.
- The datastore root is `<butlerRoot>`-relative and `copy` mode puts files under it, so the
  butler directory can be moved with globus as is.
- The butler directory holds the datasets themselves. Keep it out of git, and out of
  whatever syncs this scripts directory to USDF, since globus carries it.
- `transfer-datasets` copies datasets into the RUN collections they live in and creates no
  chain, so the sqlite butler (and `main` after step 5) holds the runs but not
  `COLLECTIONS` when that is a chain. Step 5 and the counts in `run-nersc-job.sh` therefore
  read the sqlite butler by its run names (`melon_repo_runs` in `common.sh`). Define the
  chain in `main` separately if you need it.
