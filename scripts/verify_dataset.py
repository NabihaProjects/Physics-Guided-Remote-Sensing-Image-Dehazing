"""Dataset pair verification script for SateHaze1k / Haze1k and RICE remote sensing datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image, ImageFile
import numpy as np

# Allow loading valid images that might have truncated metadata
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.base_paired import normalize_stem, VALID_IMAGE_EXTS, HAZY_FOLDER_NAMES, CLEAR_FOLDER_NAMES


def check_single_pair(
    hazy_path: Path,
    clear_path: Path,
    expected_dim: Optional[Tuple[int, int]] = None,
) -> Tuple[bool, Optional[str]]:
    """Verify integrity and spatial alignment of a single hazy-clear pair."""
    try:
        with Image.open(hazy_path) as h_img:
            h_size = h_img.size  # (width, height)
            h_mode = h_img.mode
            h_np = np.array(h_img)

        with Image.open(clear_path) as c_img:
            c_size = c_img.size
            c_mode = c_img.mode
            c_np = np.array(c_img)
    except Exception as e:
        return False, f"File corrupt / unreadable: {e}"

    if h_size != c_size:
        return False, f"Spatial dimension mismatch: hazy {h_size} vs clear {c_size}"

    if expected_dim is not None:
        exp_w, exp_h = expected_dim
        if h_size != (exp_w, exp_h):
            return False, f"Expected {expected_dim}, but got {h_size}"

    # Check identical image anomaly (hazy should not be identical to clear)
    if np.array_equal(h_np, c_np):
        return False, "Hazy and clear images are numerically identical"

    return True, None


def verify_paired_folder(
    hazy_dir: Path,
    clear_dir: Path,
    expected_dim: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Verify paired images between two directories (using normalized stem matching)."""
    hazy_files: Dict[str, Path] = {}
    for f in sorted(hazy_dir.iterdir()):
        if f.suffix.lower() in VALID_IMAGE_EXTS and not f.name.startswith("."):
            key = normalize_stem(f.stem)
            hazy_files[key] = f

    clear_files: Dict[str, Path] = {}
    for f in sorted(clear_dir.iterdir()):
        if f.suffix.lower() in VALID_IMAGE_EXTS and not f.name.startswith("."):
            key = normalize_stem(f.stem)
            clear_files[key] = f

    hazy_keys = set(hazy_files.keys())
    clear_keys = set(clear_files.keys())

    matched_keys = sorted(hazy_keys & clear_keys)
    orphaned_hazy = sorted(hazy_keys - clear_keys)
    orphaned_clear = sorted(clear_keys - hazy_keys)

    corrupted_pairs = []
    for k in matched_keys:
        valid, msg = check_single_pair(hazy_files[k], clear_files[k], expected_dim=expected_dim)
        if not valid:
            corrupted_pairs.append({"name": k, "reason": msg})

    status = "PASS" if len(orphaned_hazy) == 0 and len(orphaned_clear) == 0 and len(corrupted_pairs) == 0 else "FAIL"

    return {
        "hazy_dir": str(hazy_dir),
        "clear_dir": str(clear_dir),
        "matched_pairs": len(matched_keys),
        "orphaned_hazy_count": len(orphaned_hazy),
        "orphaned_clear_count": len(orphaned_clear),
        "orphaned_hazy_samples": orphaned_hazy[:5],
        "orphaned_clear_samples": orphaned_clear[:5],
        "corrupted_pairs_count": len(corrupted_pairs),
        "corrupted_samples": corrupted_pairs[:5],
        "status": status,
    }


def verify_haze1k_subset(
    subset_dir: Path,
    subset_name: str,
    expected_dim: Optional[Tuple[int, int]] = (512, 512),
) -> Dict[str, Any]:
    """Verify a single Haze1k subset (thin, moderate, or thick)."""
    ds_root = subset_dir / "dataset" if (subset_dir / "dataset").is_dir() else subset_dir

    # Check for split subdirectories: train, val/valid, test
    split_names = [d.name for d in ds_root.iterdir() if d.is_dir() and d.name in ("train", "val", "valid", "test")]

    if split_names:
        splits_info = {}
        total_matched = 0
        all_passed = True

        for sp in sorted(split_names):
            sp_dir = ds_root / sp
            h_dir = None
            for name in HAZY_FOLDER_NAMES:
                if (sp_dir / name).is_dir():
                    h_dir = sp_dir / name
                    break
            c_dir = None
            for name in CLEAR_FOLDER_NAMES:
                if (sp_dir / name).is_dir():
                    c_dir = sp_dir / name
                    break

            if h_dir and c_dir:
                res = verify_paired_folder(h_dir, c_dir, expected_dim=expected_dim)
                splits_info[sp] = res
                total_matched += res["matched_pairs"]
                if res["status"] != "PASS":
                    all_passed = False
            else:
                splits_info[sp] = {"error": f"Could not find hazy/clear folders in {sp_dir}", "status": "FAIL"}
                all_passed = False

        return {
            "subset": subset_name,
            "total_pairs": total_matched,
            "splits": splits_info,
            "status": "PASS" if all_passed else "FAIL",
        }
    else:
        # Flat subset folder
        h_dir = None
        for name in HAZY_FOLDER_NAMES:
            if (ds_root / name).is_dir():
                h_dir = ds_root / name
                break
        c_dir = None
        for name in CLEAR_FOLDER_NAMES:
            if (ds_root / name).is_dir():
                c_dir = ds_root / name
                break

        if h_dir and c_dir:
            res = verify_paired_folder(h_dir, c_dir, expected_dim=expected_dim)
            return {
                "subset": subset_name,
                "total_pairs": res["matched_pairs"],
                "splits": {"all": res},
                "status": res["status"],
            }

        return {"subset": subset_name, "error": f"No valid hazy/clear structure in {ds_root}", "status": "FAIL"}


def verify_dataset(
    data_dir: Path | str,
    dataset_type: str = "auto",
    expected_dim: Optional[Tuple[int, int]] = (512, 512),
) -> Dict[str, Any]:
    """Verify remote sensing dataset pair matching and integrity."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return {"error": f"Directory not found: {data_dir}", "status": "FAIL"}

    results: Dict[str, Any] = {"path": str(data_dir), "datasets": {}, "overall_status": "PASS"}
    overall_pass = True

    # 1. Check if directory contains Haze1k, RICE_DATASET, or RRSHID
    has_haze1k = (data_dir / "Haze1k").is_dir() or "haze1k" in data_dir.name.lower() or dataset_type == "satehaze1k"
    has_rice = (data_dir / "RICE_DATASET").is_dir() or (data_dir / "RICE1").is_dir() or "rice" in data_dir.name.lower() or dataset_type == "rice1"
    has_rrshid = (data_dir / "RRSHID").is_dir() or "rrshid" in data_dir.name.lower() or dataset_type == "rrshid"

    # Case A: Haze1k
    if has_haze1k and dataset_type in ("auto", "satehaze1k"):
        haze_root = data_dir / "Haze1k" if (data_dir / "Haze1k").is_dir() else data_dir
        if (haze_root / "Haze1k").is_dir():
            haze_root = haze_root / "Haze1k"

        haze_res = {"subsets": {}, "total_pairs": 0, "status": "PASS"}
        for s in ["thin", "moderate", "thick"]:
            sub_dir = None
            for candidate in [f"Haze1k_{s}", s]:
                if (haze_root / candidate).is_dir():
                    sub_dir = haze_root / candidate
                    break
            if sub_dir:
                s_res = verify_haze1k_subset(sub_dir, s, expected_dim=expected_dim)
                haze_res["subsets"][s] = s_res
                haze_res["total_pairs"] += s_res.get("total_pairs", 0)
                if s_res.get("status") != "PASS":
                    haze_res["status"] = "FAIL"
                    overall_pass = False

        results["datasets"]["Haze1k"] = haze_res

    # Case B: RICE1 / RICE_DATASET
    if has_rice and dataset_type in ("auto", "rice1"):
        rice_root = data_dir / "RICE_DATASET" if (data_dir / "RICE_DATASET").is_dir() else data_dir
        rice1_root = rice_root / "RICE1" if (rice_root / "RICE1").is_dir() else rice_root

        # Find cloud and label folders
        h_dir = None
        for name in HAZY_FOLDER_NAMES:
            if (rice1_root / name).is_dir():
                h_dir = rice1_root / name
                break
        c_dir = None
        for name in CLEAR_FOLDER_NAMES:
            if (rice1_root / name).is_dir():
                c_dir = rice1_root / name
                break

        if h_dir and c_dir:
            rice_res = verify_paired_folder(h_dir, c_dir, expected_dim=expected_dim)
            results["datasets"]["RICE1"] = {
                "total_pairs": rice_res["matched_pairs"],
                "details": rice_res,
                "status": rice_res["status"],
            }
            if rice_res["status"] != "PASS":
                overall_pass = False
        else:
            results["datasets"]["RICE1"] = {"error": f"Could not find cloud/label folders in {rice1_root}", "status": "FAIL"}
            overall_pass = False

    # Case C: RRSHID (real-world remote sensing dataset, 256x256)
    if has_rrshid and dataset_type in ("auto", "rrshid"):
        rrshid_root = data_dir / "RRSHID" if (data_dir / "RRSHID").is_dir() else data_dir
        # If user did not override dim specifically for 512, default to native (256, 256) for RRSHID
        rrshid_dim = (256, 256) if expected_dim == (512, 512) else expected_dim

        rrshid_res = {"subsets": {}, "total_pairs": 0, "status": "PASS"}
        for s in ["thin_fog", "moderate_fog", "thick_fog"]:
            sub_dir = rrshid_root / s
            if sub_dir.is_dir():
                friendly_name = s.replace("_fog", "")
                s_res = verify_haze1k_subset(sub_dir, friendly_name, expected_dim=rrshid_dim)
                rrshid_res["subsets"][friendly_name] = s_res
                rrshid_res["total_pairs"] += s_res.get("total_pairs", 0)
                if s_res.get("status") != "PASS":
                    rrshid_res["status"] = "FAIL"
                    overall_pass = False

        results["datasets"]["RRSHID"] = rrshid_res

    # Case D: Standalone flat paired directory
    if not has_haze1k and not has_rice and not has_rrshid:
        h_dir = None
        for name in HAZY_FOLDER_NAMES:
            if (data_dir / name).is_dir():
                h_dir = data_dir / name
                break
        c_dir = None
        for name in CLEAR_FOLDER_NAMES:
            if (data_dir / name).is_dir():
                c_dir = data_dir / name
                break
        if h_dir and c_dir:
            single_res = verify_paired_folder(h_dir, c_dir, expected_dim=expected_dim)
            results["datasets"]["paired"] = single_res
            if single_res["status"] != "PASS":
                overall_pass = False

    results["overall_status"] = "PASS" if overall_pass else "FAIL"
    return results


def print_verification_report(results: Dict[str, Any]) -> None:
    """Print structured CLI report."""
    print("\n" + "=" * 76)
    print("      REMOTE SENSING DATASET PAIR INTEGRITY VERIFICATION REPORT")
    print("=" * 76)
    print(f"Dataset Path   : {results.get('path')}")
    print(f"Overall Status : {results.get('overall_status')}")
    print("-" * 76)

    for ds_name, ds_info in results.get("datasets", {}).items():
        print(f"\n📂 Dataset: {ds_name} (Status: {ds_info.get('status')})")
        if "error" in ds_info:
            print(f"   ERROR: {ds_info['error']}")
            continue

        if ds_name in ("Haze1k", "RRSHID"):
            print(f"   Total Pairs Across All Subsets: {ds_info.get('total_pairs')}")
            for sub_name, sub_info in ds_info.get("subsets", {}).items():
                print(f"   ├── Subset [{sub_name:<8}]: {sub_info.get('total_pairs', 0)} pairs (Status: {sub_info.get('status')})")
                for sp_name, sp_info in sub_info.get("splits", {}).items():
                    corrupt_c = sp_info.get('corrupted_pairs_count', 0)
                    corrupt_str = f", corrupted: {corrupt_c}" if corrupt_c > 0 else ""
                    print(f"   │   ├── {sp_name:<5}: {sp_info.get('matched_pairs')} pairs (orphaned hazy: {sp_info.get('orphaned_hazy_count')}, orphaned clear: {sp_info.get('orphaned_clear_count')}{corrupt_str})")
                    if corrupt_c > 0:
                        for c_item in sp_info.get("corrupted_samples", []):
                            print(f"   │   │   └── Issue in pair {c_item['name']}: {c_item['reason']}")
        elif ds_name == "RICE1":
            details = ds_info.get("details", {})
            print(f"   Total Pairs: {ds_info.get('total_pairs')}")
            print(f"   ├── Matched Pairs   : {details.get('matched_pairs')}")
            print(f"   ├── Orphaned Hazy   : {details.get('orphaned_hazy_count')}")
            print(f"   ├── Orphaned Clear  : {details.get('orphaned_clear_count')}")
            print(f"   └── Corrupted Pairs : {details.get('corrupted_pairs_count')}")

    print("=" * 76 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify remote sensing dehazing datasets.")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--type", type=str, default="auto", choices=["auto", "satehaze1k", "rice1", "rrshid", "paired"])
    parser.add_argument("--dim", type=int, nargs=2, default=[512, 512], help="Expected (width, height) e.g. 512 512")
    parser.add_argument("--report", type=str, default=None, help="Save report JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_dim = tuple(args.dim) if args.dim else None
    report = verify_dataset(args.data_dir, dataset_type=args.type, expected_dim=expected_dim)
    print_verification_report(report)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {args.report}")

    if report.get("overall_status") != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
