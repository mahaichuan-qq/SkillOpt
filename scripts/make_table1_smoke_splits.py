"""Create tiny smoke-test splits from materialized Table 1 datasets."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
SPLITS = ("train", "val", "test")
DEFAULT_COUNTS = {"train": 5, "val": 3, "test": 5}


JSON_DATASETS = {
    "searchqa": DATA_ROOT / "searchqa_split",
    "livemathematicianbench": DATA_ROOT / "livemathematicianbench_split",
    "spreadsheetbench": DATA_ROOT / "spreadsheetbench_split",
}

CSV_DATASETS = {
    "docvqa": DATA_ROOT / "docvqa" / "splits",
    "officeqa": DATA_ROOT / "officeqa_split",
}


def _remove_tree(path: Path) -> None:
    for root, dirs, files in os.walk(path, topdown=False, onerror=lambda exc: None):
        for name in files:
            try:
                os.chmod(Path(root) / name, 0o700)
            except OSError:
                pass
        for name in dirs:
            try:
                os.chmod(Path(root) / name, 0o700)
            except OSError:
                pass
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass

    def _onerror(func, value, exc_info):
        del exc_info
        try:
            os.chmod(value, 0o700)
        except OSError:
            pass
        func(value)

    if path.exists():
        shutil.rmtree(path, onerror=_onerror)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _sort_for_smoke(dataset: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if dataset == "livemathematicianbench":
        return sorted(rows, key=lambda r: len(str(r.get("question") or r.get("problem") or "")))
    return rows


def _copy_json_dataset(dataset: str, src: Path, dst: Path, counts: dict[str, int]) -> dict[str, int]:
    actual: dict[str, int] = {}
    for split in SPLITS:
        rows = _read_json(src / split / "items.json")
        rows = _sort_for_smoke(dataset, rows)[: counts[split]]
        _write_json(dst / split / "items.json", rows)
        actual[split] = len(rows)
    return actual


def _copy_csv_dataset(dataset: str, src: Path, dst: Path, counts: dict[str, int]) -> dict[str, int]:
    del dataset
    actual: dict[str, int] = {}
    for split in SPLITS:
        csv_files = sorted((src / split).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {src / split}")
        with csv_files[0].open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)[: counts[split]]
        out = dst / split / csv_files[0].name
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        actual[split] = len(rows)
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(DATA_ROOT / "table1_smoke"))
    parser.add_argument("--train", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument("--val", type=int, default=DEFAULT_COUNTS["val"])
    parser.add_argument("--test", type=int, default=DEFAULT_COUNTS["test"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    counts = {"train": args.train, "val": args.val, "test": args.test}
    if args.force and out_root.exists():
        _remove_tree(out_root)

    summary: dict[str, dict[str, int]] = {}
    for dataset, src in JSON_DATASETS.items():
        dst = out_root / dataset
        summary[dataset] = _copy_json_dataset(dataset, src, dst, counts)
        _write_json(dst / "split_manifest.json", {
            "benchmark": dataset,
            "smoke_source": str(src.relative_to(REPO_ROOT)),
            "counts": summary[dataset],
        })
        print(f"[smoke] {dataset}: {summary[dataset]} -> {dst}")

    for dataset, src in CSV_DATASETS.items():
        dst = out_root / dataset
        summary[dataset] = _copy_csv_dataset(dataset, src, dst, counts)
        _write_json(dst / "split_manifest.json", {
            "benchmark": dataset,
            "smoke_source": str(src.relative_to(REPO_ROOT)),
            "counts": summary[dataset],
        })
        print(f"[smoke] {dataset}: {summary[dataset]} -> {dst}")

    _write_json(out_root / "split_manifest.json", {
        "type": "table1_smoke",
        "counts_requested": counts,
        "counts_actual": summary,
    })
    print("[smoke] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
