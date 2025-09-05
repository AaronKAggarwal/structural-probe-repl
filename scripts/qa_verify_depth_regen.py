#!/usr/bin/env python3
import json, math
from pathlib import Path

EPS = 1e-6

def loadj(p: Path):
    return json.loads(p.read_text())

def cmp_scalar(name: str, a, b):
    ok = (a == b) or (a is None and b is None) or (abs(a - b) <= EPS)
    diff = None if a is None or b is None else (a - b)
    return ok, f"{name}: old={a} new={b} diff={diff:.12f}" if diff is not None else f"{name}: old={a} new={b}"

def verify_pair(old_p: Path, new_p: Path, split: str, lang: str, layer: str):
    old = loadj(old_p) if old_p.exists() else {}
    new = loadj(new_p)

    print(f"\n[{lang} {layer} {split}]")

    # 1) Scalar checks (best-effort; keys may vary)
    for k in ["uuas", "root_acc", "spearmanr_content_only"]:
        a = old.get(k, None)
        b = new.get(k, None)
        if a is not None and b is not None and isinstance(a, (int, float)) and isinstance(b, (int, float)):
            ok, msg = cmp_scalar(k, float(a), float(b))
            print(("✓ " if ok else "✗ ") + msg)
        else:
            print(f"• {k}: (not present in one of the files)")

    # 2) Per-sentence (if possible)
    # Determine N from sentence_ids if present; otherwise, from Stage-3 stats
    N = len(new.get("sentence_ids", [])) or None
    kept = new.get("kept_sentence_indices", None)
    full = new.get("root_acc_per_sentence_full", None)
    comp_new = new.get("root_acc_per_sentence", None)
    comp_old = old.get("root_acc_per_sentence", None)

    if full is not None:
        if N is None:
            # Try Stage-3 stats
            stats_dir = Path("outputs/analysis/sentence_stats") / lang
            stats_file = stats_dir / f"{split}_content_stats.jsonl"
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f:
                        N = sum(1 for _ in f)
                except Exception:
                    N = None
        finite_vals = [x for x in full if isinstance(x, (int, float)) and not math.isnan(x)]
        mean_full = sum(finite_vals) / max(1, len(finite_vals))
        len_note = f" (expected N={N})" if N is not None else ""
        ok_flag = (N is None) or (len(full) == N)
        print(("✓ " if ok_flag else "✗ ") + f"root_acc_per_sentence_full: len={len(full)}{len_note} finite_mean={mean_full:.6f}")

    if comp_new is not None and kept is not None:
        assert len(comp_new) == len(kept)
        print(f"✓ root_acc_per_sentence (new compact): len={len(comp_new)}; kept indices len={len(kept)}")

    # Old vs new (compact) mean compare (only if old compact exists & lengths match)
    if comp_old is not None and comp_new is not None and len(comp_old) == len(comp_new):
        diff = sum(abs(a - b) for a, b in zip(comp_old, comp_new)) / max(1, len(comp_new))
        print(("✓ " if diff <= 1e-9 else "✗ ") + f" compact mean abs diff={diff:.12e}")

def main():
    ROOT = Path("outputs/baselines_auto")
    langs = ["UD_Czech-PDTC", "UD_English-EWT", "UD_German-GSD", "UD_Spanish-AnCora"]
    layers = ["L5", "L6", "L7", "L8", "L9", "L10"]
    for lang in langs:
        for split, fname in [("test", "test_detailed_metrics"), ("dev", "dev_detailed_metrics")]:
            for layer in layers:
                # Compare original latest vs our regen_check run when present
                run_latest = ROOT / lang / "bert-base-multilingual-cased" / "depth" / layer / "runs" / "latest"
                run_regen = ROOT / lang / "bert-base-multilingual-cased" / "depth" / layer / "runs" / "regen_check"
                old_p = run_latest / (fname + ".json")
                new_p = run_regen / (fname + ".json")
                if new_p.exists():
                    verify_pair(old_p, new_p, split, lang, layer)
                else:
                    print(f"\n[{lang} {layer} {split}] ✗ new file missing: {new_p}")

if __name__ == "__main__":
    main()


