"""Split condition-labeled episode_*.hdf5 into train/val/test with renaming + a sidecar.

This is the FIRST step of the two-stage pick-and-place data pipeline. Every episode
moves all four table objects in two sequential stages; which object goes where is the
episode's "position condition", listed by RAW episode index in:

    <src>/first_stage_position_condition.txt

Format (4 condition blocks; header line then a comma-separated raw-index line):

    (stage1) <before object> to <after object>, (stage2) <before object> to <after object>:
    2,4,6,7,32,33,...

The four objects have a FIXED size order (small -> large), which is how the downstream
SceneDiff masks are identified (see scene_diff/scripts/extract_object_positions.py):

    green cylinder = 0,  black container = 1,  yellow cup = 2,  pink cup = 3   (SIZE SLOTS)

For each condition we turn the two "before -> after" stages into a permutation of those
size slots, ordered [s1_src, s1_dst, s2_src, s2_dst]. Example:

    (stage1) green cylinder to yellow cup, (stage2) black container to pink cup
      -> [0, 2, 1, 3]   (the order the porter arranges observation.environment_state in)

Source layout:
    <src>/episode_{raw}.hdf5

Destination layout (renamed CONSECUTIVELY from 0 per split):
    <dst>/<split>/episode_{new}.hdf5
    <dst>/<split>/conditions.json     # renamed_index -> stage permutation (the sidecar)

Split policy (per condition, in listed order): the first TRAIN_PER_COND go to train, the
next VAL_PER_COND to val, the next TEST_PER_COND to test; any remainder is reported as
UNUSED. Renamed indices are assigned per split in condition-major / listed order, so
conditions.json[k] describes episode_{k}.hdf5 in that split.

The porter (examples/port_datasets/port_dexmate_hdf5.py) reads conditions.json per split
to arrange the four size-ordered SceneDiff positions into observation.environment_state.

By default files are COPIED so raw_data stays intact. Use --mode hardlink for an instant
zero-copy alternative on the same filesystem, or --mode move to relocate originals.

Run with --apply to perform the operation; otherwise prints a dry-run plan. The sidecar
is written (and the file count verified) only with --apply.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PATTERN = re.compile(r"^episode_(\d+)\.hdf5$")

DEFAULT_SRC = Path("/home/yixuan/Dexmate/data/raw_data")
DEFAULT_DST = Path("/home/yixuan/Dexmate/data/raw_data_renamed")
DEFAULT_CONDITIONS = DEFAULT_SRC / "first_stage_position_condition.txt"

# Per-condition split sizes (applied in the txt's listed order). 15 + 1 = 16 uses every
# index listed per condition when 16 are present; any extra listed indices are UNUSED.
TRAIN_PER_COND = 15
VAL_PER_COND = 1
SPLIT_SPEC = (("train", TRAIN_PER_COND), ("val", VAL_PER_COND))

# Fixed object -> SIZE SLOT map (small -> large). The keys are matched case-insensitively
# with internal whitespace collapsed. These slot ids index the size-ordered positions the
# downstream extractor writes; the porter reorders them per the stage permutation below.
OBJECT_SLOTS = {
    "green cylinder": 0,
    "black container": 1,
    "yellow cup": 2,
    "pink cup": 3,
}

# Header like "(stage1) green cylinder to yellow cup, (stage2) black container to pink cup:"
HEADER_RE = re.compile(
    r"^\(\s*stage\s*1\s*\)\s*(?P<s1>.+?)\s*,\s*\(\s*stage\s*2\s*\)\s*(?P<s2>.+?)\s*:\s*$",
    re.IGNORECASE,
)


def collect_episodes(directory: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in directory.iterdir():
        if not p.is_file():
            continue
        m = PATTERN.match(p.name)
        if not m:
            continue
        out[int(m.group(1))] = p
    return out


def normalize_object(name: str) -> str:
    """Lowercase + collapse internal whitespace so 'Yellow  Cup' -> 'yellow cup'."""
    return re.sub(r"\s+", " ", name.strip().lower())


def clause_to_slots(clause: str, source: str) -> tuple[int, int]:
    """Parse one 'before object to after object' clause to (src_slot, dst_slot)."""
    parts = re.split(r"\s+to\s+", clause.strip())
    if len(parts) != 2:
        raise ValueError(
            f"{source}: cannot parse stage clause {clause!r}; expected "
            "'<before object> to <after object>'"
        )
    slots = []
    for role, raw_name in zip(("before", "after"), parts, strict=False):
        name = normalize_object(raw_name)
        if name not in OBJECT_SLOTS:
            raise ValueError(
                f"{source}: unknown {role} object {raw_name!r} (normalized {name!r}); "
                f"expected one of {sorted(OBJECT_SLOTS)}"
            )
        slots.append(OBJECT_SLOTS[name])
    return slots[0], slots[1]


def parse_conditions_file(path: Path) -> list[dict]:
    """Parse the condition txt into a list of condition dicts (in file order).

    Each dict: {label, description, permutation [s1src,s1dst,s2src,s2dst],
    stage1 {src,dst}, stage2 {src,dst}, raw_indices [int, ... in listed order]}.
    """
    if not path.is_file():
        raise FileNotFoundError(f"condition file not found: {path}")
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    lines = [ln for ln in lines if ln]  # drop blank lines
    if len(lines) % 2 != 0:
        raise ValueError(
            f"{path}: expected alternating header/index lines (even count), got {len(lines)}"
        )

    conditions: list[dict] = []
    for ci, (header, idx_line) in enumerate(zip(lines[0::2], lines[1::2], strict=False)):
        m = HEADER_RE.match(header)
        if not m:
            raise ValueError(
                f"{path}: line {2 * ci + 1} is not a valid condition header: {header!r}"
            )
        s1_src, s1_dst = clause_to_slots(m.group("s1"), str(path))
        s2_src, s2_dst = clause_to_slots(m.group("s2"), str(path))
        try:
            raw_indices = [int(tok) for tok in idx_line.split(",") if tok.strip() != ""]
        except ValueError as exc:
            raise ValueError(
                f"{path}: line {2 * ci + 2} has a non-integer index: {idx_line!r}"
            ) from exc
        if not raw_indices:
            raise ValueError(f"{path}: condition {header!r} has no episode indices")
        dup = {i for i in raw_indices if raw_indices.count(i) > 1}
        if dup:
            raise ValueError(f"{path}: condition {header!r} repeats indices {sorted(dup)}")
        conditions.append(
            {
                "label": chr(ord("A") + ci),
                "description": re.sub(r"\s*:\s*$", "", header),
                "permutation": [s1_src, s1_dst, s2_src, s2_dst],
                "stage1": {"src": s1_src, "dst": s1_dst},
                "stage2": {"src": s2_src, "dst": s2_dst},
                "raw_indices": raw_indices,
            }
        )

    # No raw episode may belong to two conditions.
    seen: dict[int, str] = {}
    for cond in conditions:
        for i in cond["raw_indices"]:
            if i in seen:
                raise ValueError(
                    f"{path}: episode {i} listed under both condition {seen[i]} and {cond['label']}"
                )
            seen[i] = cond["label"]
    return conditions


def build_plan(
    conditions: list[dict], source_files: dict[int, Path]
) -> tuple[list[dict], dict[str, list[int]], dict[str, list[int]]]:
    """Assign each condition's available episodes to splits and renamed indices.

    Returns (entries, unused_by_cond, missing_by_cond). Each entry gains a split and a
    renamed_index. Raises if a condition can't fill its train/val quota.
    """
    entries: list[dict] = []
    unused_by_cond: dict[str, list[int]] = {}
    missing_by_cond: dict[str, list[int]] = {}

    for cond in conditions:
        available = [i for i in cond["raw_indices"] if i in source_files]
        missing = [i for i in cond["raw_indices"] if i not in source_files]
        if missing:
            missing_by_cond[cond["label"]] = missing
        needed = sum(c for _, c in SPLIT_SPEC)
        if len(available) < needed:
            raise SystemExit(
                f"ERROR: condition {cond['label']} ({cond['description']}) has only "
                f"{len(available)} available episode(s) but needs {needed} "
                f"({'/'.join(f'{n}:{c}' for n, c in SPLIT_SPEC)}); missing {missing}"
            )
        cursor = 0
        for split, count in SPLIT_SPEC:
            for raw in available[cursor : cursor + count]:
                entries.append(
                    {
                        "raw_index": raw,
                        "split": split,
                        "condition": cond["label"],
                        "description": cond["description"],
                        "permutation": cond["permutation"],
                        "stage1": cond["stage1"],
                        "stage2": cond["stage2"],
                    }
                )
            cursor += count
        leftover = available[cursor:]
        if leftover:
            unused_by_cond[cond["label"]] = leftover

    # Assign consecutive renamed indices per split (condition-major / listed order).
    for split, _ in SPLIT_SPEC:
        split_entries = [e for e in entries if e["split"] == split]
        for new_idx, e in enumerate(split_entries):
            e["renamed_index"] = new_idx
            e["src"] = source_files[e["raw_index"]]
    return entries, unused_by_cond, missing_by_cond


def conditions_payload(entries: list[dict], split: str) -> dict:
    """Build the conditions.json content for one split (renamed-index ordered)."""
    split_entries = sorted(
        (e for e in entries if e["split"] == split), key=lambda e: e["renamed_index"]
    )
    episodes = [
        {
            "renamed_index": e["renamed_index"],
            "raw_index": e["raw_index"],
            "condition": e["condition"],
            "description": e["description"],
            "order": e["permutation"],
            "stage1": e["stage1"],
            "stage2": e["stage2"],
        }
        for e in split_entries
    ]
    return {
        "object_slots": OBJECT_SLOTS,
        "order_format": "[s1_src, s1_dst, s2_src, s2_dst] as object size-slot indices",
        "num_episodes": len(episodes),
        "episodes": episodes,
    }


def transfer(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help=f"source directory of episode_*.hdf5 (default: {DEFAULT_SRC})",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_DST,
        help=f"destination root directory (default: {DEFAULT_DST})",
    )
    ap.add_argument(
        "--conditions",
        type=Path,
        default=DEFAULT_CONDITIONS,
        help=f"position-condition txt (default: {DEFAULT_CONDITIONS})",
    )
    ap.add_argument(
        "--mode",
        choices=("copy", "hardlink", "move"),
        default="copy",
        help="how to materialize files in dst (default: copy)",
    )
    ap.add_argument(
        "--apply", action="store_true", help="actually perform operations; otherwise dry-run"
    )
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"ERROR: source directory not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    source_files = collect_episodes(args.src)
    conditions = parse_conditions_file(args.conditions)
    print(f"Source: {args.src}  ({len(source_files)} episode files)")
    print(f"Conditions: {args.conditions}  ({len(conditions)} conditions)\n")

    entries, unused_by_cond, missing_by_cond = build_plan(conditions, source_files)

    # ---- Per-condition report (lets you verify the parse + split) ----
    for cond in conditions:
        label = cond["label"]
        print(f"[{label}] {cond['description']}")
        print(f"      permutation [s1_src,s1_dst,s2_src,s2_dst] = {cond['permutation']}")
        for split, _ in SPLIT_SPEC:
            raws = [
                e["raw_index"] for e in entries if e["condition"] == label and e["split"] == split
            ]
            print(f"      {split:5s}: {raws}")
        if label in unused_by_cond:
            print(f"      UNUSED: {unused_by_cond[label]}")
        if label in missing_by_cond:
            print(f"      MISSING (listed but no file): {missing_by_cond[label]}")
        print()

    # ---- Rename plan + split totals ----
    for split, _ in SPLIT_SPEC:
        split_entries = sorted(
            (e for e in entries if e["split"] == split), key=lambda e: e["renamed_index"]
        )
        print(f"[{split}] {len(split_entries)} episode(s):")
        for e in split_entries:
            print(
                f"  episode_{e['raw_index']}.hdf5  ->  {split}/episode_{e['renamed_index']}.hdf5"
                f"  (cond {e['condition']}, order {e['permutation']})"
            )
    total = len(entries)
    used = sorted(e["raw_index"] for e in entries)
    unused_total = sorted(set(source_files) - set(used))
    print(f"\nTotal episodes placed: {total}")
    print(f"Raw episodes NOT placed in any split: {unused_total}")

    if not args.apply:
        # Show what the sidecar will look like for the train split.
        preview = conditions_payload(entries, "train")
        print("\nconditions.json (train) preview — first 3 episodes:")
        print(json.dumps({**preview, "episodes": preview["episodes"][:3]}, indent=2))
        print("\nDry run. Re-run with --apply to materialize files + sidecars.")
        return

    # Pre-flight: refuse to overwrite any destination file.
    for e in entries:
        e["dst"] = args.dst / e["split"] / f"episode_{e['renamed_index']}.hdf5"
        if e["dst"].exists():
            raise SystemExit(f"ERROR: destination already exists: {e['dst']}")

    for split, _ in SPLIT_SPEC:
        split_dir = args.dst / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_entries = [e for e in entries if e["split"] == split]
        for e in split_entries:
            transfer(e["src"], e["dst"], args.mode)
        # Write the per-split conditions sidecar consumed by the porter.
        sidecar = split_dir / "conditions.json"
        sidecar.write_text(json.dumps(conditions_payload(entries, split), indent=2))
        print(f"[{split}] wrote {len(split_entries)} files + {sidecar.name} to {split_dir}")

    # ---- Verification ----
    print("\nVerification:")
    ok = True
    total_actual = 0
    for split, _ in SPLIT_SPEC:
        split_dir = args.dst / split
        actual_files = sorted(p for p in split_dir.iterdir() if PATTERN.match(p.name))
        actual = len(actual_files)
        expected = len([e for e in entries if e["split"] == split])
        total_actual += actual
        idx_set = {int(PATTERN.match(p.name).group(1)) for p in actual_files}
        contiguous = idx_set == set(range(expected))
        sidecar = split_dir / "conditions.json"
        payload = json.loads(sidecar.read_text())
        sidecar_idx = {ep["renamed_index"] for ep in payload["episodes"]}
        sidecar_ok = (
            len(payload["episodes"]) == expected
            and sidecar_idx == set(range(expected))
            and all(len(ep["order"]) == 4 for ep in payload["episodes"])
        )
        status = "OK" if (actual == expected and contiguous and sidecar_ok) else "MISMATCH"
        if status != "OK":
            ok = False
        print(
            f"  {split}: files {actual}/{expected}, contiguous={contiguous}, "
            f"sidecar_ok={sidecar_ok} -> {status}"
        )
    print(f"  total destination files: {total_actual}; total planned: {total}")
    if total_actual != total:
        ok = False
    if not ok:
        sys.exit(1)
    print("All splits verified.")


if __name__ == "__main__":
    main()
