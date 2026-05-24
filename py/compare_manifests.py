from __future__ import annotations

import argparse
from pathlib import Path

from _qwen_paths import load_json, write_json


def compare_manifests(source: dict, export: dict) -> dict:
    source_entries = {entry["path"]: entry for entry in source["entries"]}
    export_entries = {entry["path"]: entry for entry in export["entries"]}

    source_keys = set(source_entries)
    export_keys = set(export_entries)

    missing_in_export = sorted(source_keys - export_keys)
    missing_in_source = sorted(export_keys - source_keys)
    mismatches: list[dict[str, str]] = []

    for path in sorted(source_keys & export_keys):
        left = source_entries[path]
        right = export_entries[path]
        if left["shape"] != right["shape"]:
            mismatches.append(
                {
                    "path": path,
                    "reason": f"shape mismatch: source={left['shape']} export={right['shape']}",
                }
            )
            continue
        if left["dtype"] != right["dtype"]:
            mismatches.append(
                {
                    "path": path,
                    "reason": f"dtype mismatch: source={left['dtype']} export={right['dtype']}",
                }
            )
            continue
        if left["sha256"] != right["sha256"]:
            mismatches.append(
                {
                    "path": path,
                    "reason": f"sha256 mismatch: source={left['sha256']} export={right['sha256']}",
                }
            )

    return {
        "exact_match": not (missing_in_export or missing_in_source or mismatches),
        "tensor_count": len(source_entries),
        "source_tensor_count": len(source_entries),
        "export_tensor_count": len(export_entries),
        "missing_in_export": missing_in_export,
        "missing_in_source": missing_in_source,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("export_manifest", type=Path)
    parser.add_argument("report_path", type=Path)
    args = parser.parse_args()

    source_manifest = load_json(args.source_manifest)
    export_manifest = load_json(args.export_manifest)
    report = compare_manifests(source_manifest, export_manifest)
    write_json(args.report_path, report)

    print(f"source_manifest: {args.source_manifest}")
    print(f"export_manifest: {args.export_manifest}")
    print(f"report: {args.report_path}")
    print(f"exact_match: {report['exact_match']}")
    print(f"tensor_count: {report['tensor_count']}")

    if not report["exact_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
