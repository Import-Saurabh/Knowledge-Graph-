#!/usr/bin/env python3
"""
reset_db.py  —  Wipe ALL persisted data and start fresh.

What this script does:
  1. Drops + recreates every SQLite table via get_engine() + Base
  2. Deletes and recreates the three ChromaDB collections
  3. Removes data/exports/* and data/processed/* output files
  4. Optionally removes data/raw/* downloaded articles

Usage:
  python reset_db.py              # wipe DB + Chroma + exports (keep raw)
  python reset_db.py --all        # also wipe data/raw/
  python reset_db.py --dry-run    # print what WOULD be deleted, touch nothing

After reset, run a fresh pipeline:
  python main.py --run-all --download \
      --topic "Iran war 2026" \
      --start-date 2026-02-01 --end-date 2026-06-16 \
      --interval-days 3 --max-per-interval 50 \
      --llm-batch-size 10 --llm-workers 4

Load into Neo4j after pipeline completes:
  neo4j-admin database import full \
      --nodes=data/exports/nodes.csv \
      --relationships=data/exports/relationships.csv \
      --overwrite-destination --database=neo4j
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _header(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def reset_sqlite(dry_run: bool):
    _header("Resetting SQLite database")
    try:
        # FIX: db.py exposes get_engine() + Base, NOT a bare `engine` variable
        from src.utils.db import get_engine, Base

        if dry_run:
            print("  [dry-run] would drop and recreate all SQLite tables via get_engine()")
            tables = list(Base.metadata.tables.keys())
            print(f"  Tables that would be dropped ({len(tables)}): {', '.join(tables)}")
            return

        engine = get_engine()
        print("  Dropping all tables ...")
        Base.metadata.drop_all(bind=engine)
        print("  Recreating all tables ...")
        Base.metadata.create_all(bind=engine)
        print("  ✓ SQLite reset complete")

    except Exception as e:
        print(f"  ✗ SQLite reset failed: {e}")
        raise


def reset_chroma(dry_run: bool):
    _header("Resetting ChromaDB collections")
    try:
        from src.utils.config import settings
        import chromadb

        persist_dir  = settings.CHROMA_PERSIST_DIR
        collections  = ["news_articles", "canonical_entities", "relation_ontology"]

        if dry_run:
            print(f"  [dry-run] would delete + recreate in: {persist_dir}")
            for c in collections:
                print(f"    • {c}")
            return

        client = chromadb.PersistentClient(path=persist_dir)
        for name in collections:
            try:
                client.delete_collection(name)
                print(f"  deleted : {name}")
            except Exception:
                print(f"  (skip)  : '{name}' did not exist")
            client.get_or_create_collection(name)
            print(f"  created : {name}")

        print("  ✓ ChromaDB reset complete")

    except Exception as e:
        print(f"  ✗ ChromaDB reset failed: {e}")
        raise


def reset_files(dry_run: bool, wipe_raw: bool):
    _header("Removing output / processed files")
    dirs_to_clean = ["data/exports", "data/processed"]
    if wipe_raw:
        dirs_to_clean.append("data/raw")

    for d in dirs_to_clean:
        if not os.path.exists(d):
            print(f"  (skip) {d} does not exist")
            continue
        if dry_run:
            items = os.listdir(d)
            print(f"  [dry-run] would delete {len(items)} item(s) from {d}/")
            continue
        shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        print(f"  ✓ cleared {d}/")


def main():
    parser = argparse.ArgumentParser(
        description="Wipe all news_kg persisted data for a fresh pipeline run."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Also wipe data/raw/ (downloaded articles). Re-download required after this.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be deleted without actually deleting anything.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("\n⚠  DRY-RUN MODE — nothing will be modified\n")

    reset_sqlite(dry_run=args.dry_run)
    reset_chroma(dry_run=args.dry_run)
    reset_files(dry_run=args.dry_run, wipe_raw=args.all)

    if not args.dry_run:
        print("\n✅  All data wiped. Run the pipeline with:\n")
        print(
            "  python main.py --run-all --download \\\n"
            '      --topic "Iran war 2026" \\\n'
            "      --start-date 2026-02-01 --end-date 2026-06-16 \\\n"
            "      --interval-days 3 --max-per-interval 50 \\\n"
            "      --llm-batch-size 10 --llm-workers 4\n"
        )
        print("  Then load into Neo4j:\n")
        print(
            "  neo4j-admin database import full \\\n"
            "      --nodes=data/exports/nodes.csv \\\n"
            "      --relationships=data/exports/relationships.csv \\\n"
            "      --overwrite-destination --database=neo4j\n"
        )
    else:
        print("\n✅  Dry-run complete. Re-run without --dry-run to apply.\n")


if __name__ == "__main__":
    main()