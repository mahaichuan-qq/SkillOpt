"""Materialize the five SkillOpt Table 1 datasets from offline raw files.

This script only prepares data files. It does not change SkillOpt training,
rollout, evaluation, or model backend code.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
SPLITS = ("train", "val", "test")
sys.path.insert(0, str(REPO_ROOT))


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def _split_manifest(name: str) -> dict[str, Any]:
    return _read_json(DATA_ROOT / f"{name}_id_split" / "split_manifest.json")


def _manifest_rows(name: str, split: str) -> list[dict[str, Any]]:
    path = DATA_ROOT / f"{name}_id_split" / split / "items.json"
    return _read_json(path)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return [value]


def _normalise_searchqa_row(row: dict[str, Any]) -> dict[str, Any]:
    answers = [str(x) for x in _as_list(row.get("answers")) if str(x).strip()]
    return {
        "id": str(row.get("key") or row.get("id") or "").strip(),
        "key": str(row.get("key") or "").strip(),
        "question": str(row.get("question") or "").strip(),
        "context": str(row.get("context") or "").strip(),
        "answers": answers,
        "answer": answers[0] if answers else "",
    }


def materialize_searchqa(raw_root: Path, force: bool = False) -> None:
    import pandas as pd

    raw_dir = raw_root / "searchqa_raw"
    out_dir = DATA_ROOT / "searchqa_split"
    if force and out_dir.exists():
        _remove_tree(out_dir)

    by_id: dict[str, dict[str, Any]] = {}
    for parquet in sorted(raw_dir.glob("*.parquet")):
        print(f"[SearchQA] reading {parquet}")
        df = pd.read_parquet(parquet)
        for row in df.to_dict("records"):
            item = _normalise_searchqa_row(row)
            by_id[item["id"]] = item

    counts: dict[str, int] = {}
    for split in SPLITS:
        rows = []
        for ref in _manifest_rows("searchqa", split):
            item_id = str(ref.get("id") or ref.get("key") or "").strip()
            if item_id not in by_id:
                raise KeyError(f"SearchQA id missing from raw data: {item_id}")
            rows.append(by_id[item_id])
        _write_json(out_dir / split / "items.json", rows)
        counts[split] = len(rows)

    _write_json(out_dir / "split_manifest.json", {
        "benchmark": "SearchQA",
        "source": str(raw_dir),
        "id_manifest": "data/searchqa_id_split",
        "counts": counts,
    })
    print(f"[SearchQA] wrote {counts} -> {out_dir}")


_LIVEMATH_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]


def _livemath_normalize_label(text: Any) -> str:
    return str(text).strip().upper().rstrip(".):")


def _livemath_coerce_choices(raw_choices: Any) -> list[dict[str, str]]:
    if isinstance(raw_choices, list):
        choices: list[dict[str, str]] = []
        for idx, item in enumerate(raw_choices):
            if isinstance(item, dict):
                label = str(item.get("label") or _LIVEMATH_CHOICE_LABELS[idx]).strip()
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                label = _LIVEMATH_CHOICE_LABELS[idx]
                text = str(item).strip()
            if text:
                choices.append({"label": label, "text": text})
        return choices
    if isinstance(raw_choices, dict):
        return [
            {"label": str(label).strip(), "text": str(raw_choices[label]).strip()}
            for label in sorted(raw_choices.keys())
            if str(raw_choices[label]).strip()
        ]
    return []


def _livemath_coerce_theorem_types(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if raw is None:
        return []
    text = str(raw).strip()
    return [text] if text else []


def _normalise_livemath_item(row: dict[str, Any], row_idx: int, source_path: Path) -> dict[str, Any]:
    mcq = row.get("mcq", {}) if isinstance(row.get("mcq"), dict) else {}
    question = str(mcq.get("question") or row.get("question") or "").strip()
    choices = _livemath_coerce_choices(mcq.get("choices") or row.get("choices") or [])
    correct = mcq.get("correct_choice") or row.get("correct_choice") or {}

    if isinstance(correct, dict):
        correct_label = _livemath_normalize_label(correct.get("label", ""))
        correct_text = str(correct.get("text") or "").strip()
    else:
        correct_label = _livemath_normalize_label(correct)
        correct_text = ""

    choice_by_label = {
        _livemath_normalize_label(choice["label"]): choice["text"]
        for choice in choices
    }
    if correct_label and not correct_text:
        correct_text = choice_by_label.get(correct_label, "")
    if correct_label and correct_text and correct_label not in choice_by_label:
        choices.append({"label": correct_label, "text": correct_text})
        choices.sort(
            key=lambda choice: (
                _LIVEMATH_CHOICE_LABELS.index(choice["label"])
                if choice["label"] in _LIVEMATH_CHOICE_LABELS
                else len(_LIVEMATH_CHOICE_LABELS)
            )
        )

    month = str(row.get("month") or "").strip()
    item_no = row.get("no", row_idx + 1)
    item_id = f"{month}:{item_no}" if month else str(item_no)
    return {
        "id": item_id,
        "month": month,
        "no": item_no,
        "paper_link": str(row.get("paper_link") or "").strip(),
        "theorem": str(row.get("theorem") or "").strip(),
        "sketch": str(row.get("sketch") or "").strip(),
        "theorem_type": _livemath_coerce_theorem_types(row.get("theorem_type")),
        "question": question,
        "choices": choices,
        "correct_choice": {"label": correct_label, "text": correct_text},
        "source_path": str(source_path),
    }


def materialize_livemath(raw_root: Path, force: bool = False) -> None:

    raw_dir = raw_root / "livemathematicianbench_raw"
    out_dir = DATA_ROOT / "livemathematicianbench_split"
    if force and out_dir.exists():
        _remove_tree(out_dir)

    print(f"[LiveMathematicianBench] loading from {raw_dir}")
    by_id: dict[str, dict[str, Any]] = {}
    for path in sorted(raw_dir.glob("qa_*_final.json")):
        print(f"[LiveMathematicianBench] reading {path}")
        data = _read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
        for row_idx, row in enumerate(data):
            item = _normalise_livemath_item(row, row_idx=row_idx, source_path=path)
            if item["question"] and item["choices"] and item["correct_choice"]["label"]:
                by_id[item["id"]] = item

    counts: dict[str, int] = {}
    for split in SPLITS:
        rows = []
        for ref in _manifest_rows("livemathematicianbench", split):
            item_id = str(ref.get("id") or "").strip()
            if item_id not in by_id:
                raise KeyError(f"LiveMathematicianBench id missing from raw data: {item_id}")
            rows.append(by_id[item_id])
        _write_json(out_dir / split / "items.json", rows)
        counts[split] = len(rows)

    _write_json(out_dir / "split_manifest.json", {
        "benchmark": "LiveMathematicianBench",
        "source": str(raw_dir),
        "id_manifest": "data/livemathematicianbench_id_split",
        "counts": counts,
    })
    print(f"[LiveMathematicianBench] wrote {counts} -> {out_dir}")


def materialize_spreadsheetbench(raw_root: Path, force: bool = False) -> None:
    raw_dir = raw_root / "spreadsheetbench_raw"
    archive = raw_dir / "spreadsheetbench_verified_400.tar.gz"
    data_out = DATA_ROOT / "spreadsheetbench_verified_400"
    split_out = DATA_ROOT / "spreadsheetbench_split"
    if force and data_out.exists():
        _remove_tree(data_out)
    if force and split_out.exists():
        _remove_tree(split_out)

    if not (data_out / "dataset.json").exists():
        print(f"[SpreadsheetBench] extracting {archive}")
        _safe_extract_tar(archive, data_out, strip_prefix="spreadsheetbench_verified_400")

    dataset = _read_json(data_out / "dataset.json")
    by_id = {str(row.get("id")): row for row in dataset}

    counts: dict[str, int] = {}
    for split in SPLITS:
        rows = []
        for ref in _manifest_rows("spreadsheetbench", split):
            item_id = str(ref.get("id") or "").strip()
            if item_id not in by_id:
                raise KeyError(f"SpreadsheetBench id missing from dataset.json: {item_id}")
            rows.append(by_id[item_id])
        _write_json(split_out / split / "items.json", rows)
        counts[split] = len(rows)

    _write_json(split_out / "split_manifest.json", {
        "benchmark": "SpreadsheetBench",
        "source": str(archive),
        "data_root": "data/spreadsheetbench_verified_400",
        "id_manifest": "data/spreadsheetbench_id_split",
        "counts": counts,
    })
    print(f"[SpreadsheetBench] wrote {counts} -> {split_out}")


def _save_docvqa_image(raw_image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = raw_image
    if isinstance(raw_image, dict):
        data = raw_image.get("bytes") or raw_image.get("data") or raw_image.get("path")
    if isinstance(data, (bytes, bytearray)):
        path.write_bytes(bytes(data))
        return
    try:
        from PIL import Image

        if hasattr(data, "save"):
            data.save(path)
            return
        img = Image.open(io.BytesIO(bytes(data)))
        img.save(path)
        return
    except Exception as exc:
        raise ValueError(f"Cannot save DocVQA image to {path}") from exc


def _normalise_docvqa_row(row: dict[str, Any], image_path: str) -> dict[str, Any]:
    answers = [str(x) for x in _as_list(row.get("answers")) if str(x).strip()]
    return {
        "questionId": str(row.get("questionId") or "").strip(),
        "id": str(row.get("questionId") or "").strip(),
        "docId": str(row.get("docId") or "").strip(),
        "ucsf_document_id": str(row.get("ucsf_document_id") or "").strip(),
        "ucsf_document_page_no": str(row.get("ucsf_document_page_no") or "").strip(),
        "question": str(row.get("question") or "").strip(),
        "answer": repr(answers),
        "ground_truth": repr(answers),
        "image_path": image_path,
        "topic": ";".join(str(x) for x in _as_list(row.get("question_types"))),
        "source_split": str(row.get("data_split") or "").strip(),
    }


def materialize_docvqa(raw_root: Path, force: bool = False) -> None:
    import pandas as pd

    raw_dir = raw_root / "docvqa_raw"
    split_out = DATA_ROOT / "docvqa" / "splits"
    image_root = DATA_ROOT / "docvqa_images"
    if force and split_out.exists():
        _remove_tree(split_out)
    if force and image_root.exists():
        _remove_tree(image_root)

    wanted: dict[str, str] = {}
    for split in SPLITS:
        for ref in _manifest_rows("docvqa", split):
            wanted[str(ref.get("id") or ref.get("questionId") or "").strip()] = split

    rows_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    seen: set[str] = set()
    for parquet in sorted(raw_dir.glob("*.parquet")):
        print(f"[DocVQA] reading {parquet}")
        df = pd.read_parquet(parquet)
        for row in df.to_dict("records"):
            qid = str(row.get("questionId") or "").strip()
            split = wanted.get(qid)
            if not split:
                continue
            rel_image = f"data/docvqa_images/q{qid}.png"
            _save_docvqa_image(row.get("image"), REPO_ROOT / rel_image)
            rows_by_split[split].append(_normalise_docvqa_row(row, rel_image))
            seen.add(qid)

    missing = sorted(set(wanted) - seen)
    if missing:
        raise KeyError(f"DocVQA ids missing from raw data, first 10: {missing[:10]}")

    fieldnames = [
        "questionId", "id", "docId", "ucsf_document_id", "ucsf_document_page_no",
        "question", "answer", "ground_truth", "image_path", "topic", "source_split",
    ]
    counts: dict[str, int] = {}
    for split, rows in rows_by_split.items():
        _write_csv(split_out / split / "items.csv", rows, fieldnames)
        counts[split] = len(rows)

    _write_json(split_out / "split_manifest.json", {
        "benchmark": "DocVQA",
        "source": str(raw_dir),
        "image_root": "data/docvqa_images",
        "id_manifest": "data/docvqa_id_split",
        "counts": counts,
    })
    print(f"[DocVQA] wrote {counts} -> {split_out}")


def _copy_tree(src: Path, dst: Path, force: bool = False) -> None:
    if force and dst.exists():
        _remove_tree(dst)
    if not dst.exists():
        shutil.copytree(src, dst)


def _safe_extract_tar(archive: Path, dst: Path, strip_prefix: str = "") -> None:
    """Extract a trusted dataset tarball without applying archived permissions."""
    with tarfile.open(archive, "r:gz") as t:
        for member in t.getmembers():
            member_name = member.name
            if strip_prefix:
                prefix = strip_prefix.rstrip("/") + "/"
                if member_name == strip_prefix:
                    continue
                if member_name.startswith(prefix):
                    member_name = member_name[len(prefix):]
                else:
                    continue
            if not member_name:
                continue
            target = (dst / member_name).resolve()
            if not str(target).startswith(str(dst.resolve())):
                raise ValueError(f"Unsafe tar member path: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = t.extractfile(member)
            if src is None:
                continue
            with src, target.open("wb") as f:
                shutil.copyfileobj(src, f)


def materialize_officeqa(raw_root: Path, force: bool = False) -> None:
    raw_dir = raw_root / "officeqa_hf"
    if not raw_dir.exists():
        raw_dir = raw_root / "officeqa_raw"
    csv_path = raw_dir / "officeqa_full.csv"
    docs_src = raw_dir / "treasury_bulletins_parsed"
    docs_dst = DATA_ROOT / "officeqa_docs_official" / "treasury_bulletins_parsed"
    split_out = DATA_ROOT / "officeqa_split"
    raw_copy = DATA_ROOT / "officeqa_raw"
    if force and split_out.exists():
        _remove_tree(split_out)
    if force and raw_copy.exists():
        _remove_tree(raw_copy)

    raw_copy.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_path, raw_copy / "officeqa_full.csv")
    if docs_src.exists():
        _copy_tree(docs_src, docs_dst, force=force)

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        source_fields = list(reader.fieldnames or [])
        rows = list(reader)
    by_id = {str(row.get("uid") or row.get("id") or "").strip(): row for row in rows}
    extra_fields = ["source_files", "source_docs", "category", "split"]
    fieldnames = list(dict.fromkeys(source_fields + extra_fields))

    counts: dict[str, int] = {}
    for split in SPLITS:
        out_rows = []
        for ref in _manifest_rows("officeqa", split):
            item_id = str(ref.get("id") or ref.get("uid") or "").strip()
            if item_id not in by_id:
                raise KeyError(f"OfficeQA id missing from officeqa_full.csv: {item_id}")
            row = dict(by_id[item_id])
            for key in extra_fields:
                if key in ref and not row.get(key):
                    value = ref[key]
                    row[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
            out_rows.append(row)
        _write_csv(split_out / split / "items.csv", out_rows, fieldnames)
        counts[split] = len(out_rows)

    _write_json(split_out / "split_manifest.json", {
        "benchmark": "OfficeQA",
        "source": str(csv_path),
        "docs": "data/officeqa_docs_official",
        "id_manifest": "data/officeqa_id_split",
        "counts": counts,
    })
    print(f"[OfficeQA] wrote {counts} -> {split_out}")


MATERIALIZERS = {
    "searchqa": materialize_searchqa,
    "livemathematicianbench": materialize_livemath,
    "spreadsheetbench": materialize_spreadsheetbench,
    "docvqa": materialize_docvqa,
    "officeqa": materialize_officeqa,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=str(REPO_ROOT.parent / "benchmark"))
    parser.add_argument("--dataset", choices=sorted(MATERIALIZERS), action="append")
    parser.add_argument("--force", action="store_true", help="overwrite existing materialized outputs")
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    datasets = args.dataset or list(MATERIALIZERS)
    print(f"[table1] raw_root={raw_root}")
    for name in datasets:
        MATERIALIZERS[name](raw_root, force=args.force)
    print("[table1] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
