#!/usr/bin/env python
"""Step 5b (run at USDF): put the sqlite butler's datasets into main under a run you can write.

Use this instead of 05_transfer_to_main.sh when the runs belong to another user's namespace
(e.g. u/descdm/...). transfer-datasets keeps each file's path, so it would write under
main's u/descdm/, which you can't create. This script ingests the files into a renamed run
instead (u/descdm/X -> u/$USER/X by default), and main lays them out under that new name.

    ./05b_ingest_renamed.py /sdf/data/rubin/repo/main_20210215/DECam/endurance/chunk0_template --dry-run
    ./05b_ingest_renamed.py /sdf/data/rubin/repo/main_20210215/DECam/endurance/chunk0_template

Notes:
- The default transfer is hardlink. The sqlite butler is on the same filesystem as main's
  datastore, so this uses no extra space, and both butlers stay valid. Because they are
  hardlinks, deleting the sqlite butler later leaves main's files intact.
- New dataset IDs are derived from (dataset type, data ID, new run), so they differ from
  the NERSC IDs but are the same on every run of this script. Rerunning after an
  interruption skips what is already in main.
- Ingest runs in batches, each in its own transaction.
- Defaults for --dataset-type, --where and --dest follow DATASET_TYPE, WHERE and MAIN_REPO,
  as in common.sh.
"""

import argparse
import os
import sys

from lsst.daf.butler import (
    Butler,
    CollectionType,
    DatasetIdGenEnum,
    DatasetRef,
    FileDataset,
    MissingCollectionError,
    MissingDatasetTypeError,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="path to the sqlite butler (directory or its butler.yaml)")
    p.add_argument("--collections", default="*",
                   help="comma-separated names/globs of RUN collections in the sqlite butler [*]")
    p.add_argument("--to-prefix", default=f"u/{os.environ.get('USER', '')}",
                   help="namespace for the new runs [u/$USER]")
    p.add_argument("--from-prefix", default=None,
                   help="prefix to replace in each run name [the run's own u/<user>]")
    p.add_argument("--dest", default=os.environ.get("MAIN_REPO", "main"), help="destination repo [main]")
    p.add_argument("--dataset-type", default=os.environ.get("DATASET_TYPE", "template_coadd"))
    p.add_argument("--where", default=os.environ.get("WHERE", "skymap='lsst_cells_v2'"))
    p.add_argument("--transfer", default="hardlink", choices=["hardlink", "copy", "move", "symlink"],
                   help="how files get into main's datastore [hardlink]")
    p.add_argument("--batch-size", type=int, default=5000, help="datasets per ingest transaction [5000]")
    p.add_argument("--limit", type=int, default=None, help="only the first N datasets per run (testing)")
    p.add_argument("--dry-run", action="store_true", help="report what would happen; write nothing")
    return p.parse_args()


def renamed(run: str, from_prefix: str | None, to_prefix: str) -> str:
    if from_prefix is None:
        parts = run.split("/")
        if len(parts) < 3 or parts[0] != "u":
            sys.exit(f"ERROR: run {run!r} is not under u/<user>/; pass --from-prefix.")
        from_prefix = "/".join(parts[:2])
    from_prefix, to_prefix = from_prefix.rstrip("/"), to_prefix.rstrip("/")
    if not run.startswith(from_prefix + "/"):
        sys.exit(f"ERROR: run {run!r} does not start with {from_prefix!r}/.")
    return to_prefix + run[len(from_prefix):]


def check_writable(dest: Butler, run: str) -> None:
    """Fail early if we can't create <datastore root>/<run> in the destination."""
    try:
        root = dest._datastore.root.ospath
    except Exception as e:  # e.g. a non-POSIX or chained datastore
        print(f"WARNING: skipping write-access check: {e}", file=sys.stderr)
        return
    d = os.path.join(root, run)
    while not os.path.exists(d):
        d = os.path.dirname(d)
    if not os.access(d, os.W_OK):
        sys.exit(f"ERROR: {os.path.join(root, run)} would be created under {d}, which you cannot write.")


def main() -> None:
    args = parse_args()
    if not args.to_prefix.strip("/") or args.to_prefix.rstrip("/") == "u":
        sys.exit("ERROR: --to-prefix is empty (is $USER set?).")

    src = Butler.from_config(args.source, writeable=False)
    dest = Butler.from_config(args.dest, writeable=not args.dry_run)

    runs = src.collections.query(args.collections.split(","), collection_types=CollectionType.RUN)
    if not runs:
        sys.exit(f"ERROR: no RUN collections in {args.source} match {args.collections!r}.")
    plan = {run: renamed(run, args.from_prefix, args.to_prefix) for run in runs}

    src_type = src.get_dataset_type(args.dataset_type)
    try:
        dest_type = dest.get_dataset_type(args.dataset_type)
    except MissingDatasetTypeError:
        print(f"{args.dataset_type} is not registered in {args.dest}; will register {src_type}.")
        dest_type = src_type
    if dest_type != src_type:
        sys.exit(f"ERROR: {args.dataset_type} differs between repos: {src_type} vs {dest_type}.")

    for run, new_run in plan.items():
        print(f"\n{run}\n  -> {new_run}")
        check_writable(dest, new_run)

        refs = src.query_datasets(
            src_type, collections=run, where=args.where, with_dimension_records=True,
            limit=args.limit, explain=False,
        )
        if not refs:
            print("  no matching datasets; skipping.")
            continue
        uris = src.get_many_uris(refs)
        datasets = []
        for ref in refs:
            ref_uris = uris[ref]
            if ref_uris.componentURIs:
                sys.exit(f"ERROR: {ref} is stored disassembled; this script handles single files only.")
            new_ref = DatasetRef(
                dest_type, ref.dataId, run=new_run, id_generation_mode=DatasetIdGenEnum.DATAID_TYPE_RUN
            )
            datasets.append(FileDataset(path=ref_uris.primaryURI, refs=[new_ref]))

        try:
            if dest.collections.get_info(new_run).type != CollectionType.RUN:
                sys.exit(f"ERROR: {new_run} exists in {args.dest} but is not a RUN collection.")
            n_before = len(dest.query_datasets(dest_type, collections=new_run, limit=None, explain=False))
        except MissingCollectionError:
            n_before = 0
        print(f"  {len(datasets)} datasets in source; {n_before} already in {args.dest}:{new_run}")
        print(f"  e.g. {datasets[0].path}")

        if args.dry_run:
            print(f"  dry run: would {args.transfer} {len(datasets)} files into {args.dest}.")
            continue

        dest.registry.registerDatasetType(dest_type)  # no-op if already registered identically
        dest.collections.register(new_run)
        for i in range(0, len(datasets), args.batch_size):
            batch = datasets[i : i + args.batch_size]
            dest.ingest(*batch, transfer=args.transfer, skip_existing=True)
            print(f"  ingested {min(i + len(batch), len(datasets))}/{len(datasets)}", flush=True)

        n_after = len(dest.query_datasets(dest_type, collections=new_run, limit=None, explain=False))
        print(f"  {args.dest}:{new_run} now has {n_after} {args.dataset_type} (source has {len(datasets)}).")


if __name__ == "__main__":
    main()
