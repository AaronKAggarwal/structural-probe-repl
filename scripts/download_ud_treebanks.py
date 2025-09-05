#!/usr/bin/env python3
"""
Download UD treebanks, parse stats.xml, and log metadata.

Inputs:
  - A CSV/TSV file listing UD GitHub repos to fetch with optional ref/tag/commit.
    Required column: url
    Optional columns: language, treebank, ref

Actions per row:
  1) Clone the repo into a temporary directory (shallow by default).
  2) Checkout the specified ref (if provided).
  3) Parse stats.xml and extract useful metadata (sizes, inventories, morphology proxies).
  4) Locate train/dev/test .conllu files and copy them to data/ud/<treebank_slug>/
     as train.conllu, dev.conllu, test.conllu. Also copy stats.xml for record.
  5) Write/append a consolidated CSV at data/lang_stats/ud_metadata.csv.

Notes:
  - No external Python dependencies; uses subprocess to drive git.
  - Robust to minor variations in UD repo structure by searching for files.
  - If multiple *-ud-*.conllu files are found, picks the largest by size.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


log = logging.getLogger(__name__)


@dataclass
class RepoSpec:
    url: str
    language: Optional[str]
    treebank: Optional[str]
    ref: Optional[str]


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = process.communicate()
    return process.returncode, out.strip(), err.strip()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_repo_url(url: str) -> str:
    """Normalize a GitHub URL to the repository root if the link points to a file/tree/blob.

    Examples:
      https://github.com/UniversalDependencies/UD_Russian-SynTagRus/blob/master/ru.conllu
      -> https://github.com/UniversalDependencies/UD_Russian-SynTagRus
    """
    try:
        from urllib.parse import urlparse
    except Exception:
        return url
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        # Accept UD shortnames like UD_Russian-SynTagRus
        if url.startswith("UD_"):
            return f"https://github.com/UniversalDependencies/{url}"
        return url
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2:
        owner, repo = parts[0], parts[1]
        return f"https://github.com/{owner}/{repo}"
    return url


def read_input_table(path: Path) -> List[RepoSpec]:
    delimiter = "\t" if path.suffix.lower() in {".tsv"} else ","
    specs: List[RepoSpec] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        required = {"url"}
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Input file missing required columns: {missing}. Present: {reader.fieldnames}")
        for row in reader:
            url_raw = (row.get("url") or "").strip()
            url = normalize_repo_url(url_raw)
            if not url:
                continue
            specs.append(
                RepoSpec(
                    url=url,
                    language=(row.get("language") or row.get("language_name") or None),
                    treebank=(row.get("treebank") or row.get("ud_treebank_id") or None),
                    ref=(row.get("ref") or row.get("ud_release") or None),
                )
            )
    return specs


def clone_repo(url: str, dest_dir: Path, ref: Optional[str], depth: int = 1) -> None:
    if dest_dir.exists():
        log.info(f"Clone destination exists, reusing: {dest_dir}")
    else:
        parent = dest_dir.parent
        safe_mkdir(parent)
        cmd = ["git", "clone", "--depth", str(depth), url, str(dest_dir)]
        code, out, err = run_cmd(cmd)
        if code != 0:
            raise RuntimeError(f"git clone failed for {url}: {err or out}")

    if ref:
        norm_ref = normalize_ud_ref(ref)
        # Fetch the exact ref shallowly and try checkout
        run_cmd(["git", "fetch", "--tags", "--force", "origin", norm_ref, "--depth", str(depth)], cwd=dest_dir)
        code, out, err = run_cmd(["git", "checkout", norm_ref], cwd=dest_dir)
        if code != 0:
            # Unshallow and retry
            run_cmd(["git", "fetch", "--unshallow"], cwd=dest_dir)
            code2, out2, err2 = run_cmd(["git", "checkout", norm_ref], cwd=dest_dir)
            if code2 != 0:
                raise RuntimeError(f"git checkout {norm_ref} failed in {dest_dir}: {err2 or out2}")


def detect_commit_and_tag(repo_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    code, out, err = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    commit = out if code == 0 else None
    tag: Optional[str] = None
    code, out, err = run_cmd(["git", "tag", "--points-at", "HEAD"], cwd=repo_dir)
    if code == 0 and out:
        first = out.splitlines()[0].strip()
        tag = first or None
    else:
        code2, out2, err2 = run_cmd(["git", "describe", "--tags", "--always"], cwd=repo_dir)
        if code2 == 0 and out2:
            tag = out2
    return commit, tag


def normalize_ud_ref(ref: str) -> str:
    r = ref.strip()
    # Convert various UD version formats to the actual tag format (r2.X)
    if r.startswith("r2."):
        return r  # Already in correct format
    if r.startswith("v2."):
        return r.replace("v2.", "r2.")  # Convert v2.X to r2.X
    if r.startswith("2."):
        return f"r{r}"  # Convert 2.X to r2.X
    if r.startswith("ud-treebanks-v2."):
        return r.replace("ud-treebanks-v2.", "r2.")  # Convert old format
    return r  # Use as-is for other formats


def process_repo_return_row(
    idx: int,
    total: int,
    spec: RepoSpec,
    tmp_root: Path,
    dest_root: Path,
    default_ref: Optional[str],
    strict_ref: bool,
) -> Dict[str, Optional[str]]:
    chosen_ref = spec.ref or default_ref or 'HEAD'
    log.info(f"[{idx}/{total}] Processing: {spec.url} (ref={chosen_ref})")
    row = process_repo(
        spec,
        tmp_root,
        dest_root,
        default_ref=default_ref,
        strict_ref=strict_ref,
    )
    log.info(f"  Completed: {row.get('treebank_slug')}")
    return row


def find_file_recursive(root: Path, pattern_suffix: str) -> Optional[Path]:
    candidates: List[Path] = []
    for p in root.rglob("*" + pattern_suffix):
        if p.is_file():
            candidates.append(p)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer the file with the largest size (often the main stats or main split file)
    candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
    return candidates[0]


def find_stats_xml(repo_dir: Path) -> Optional[Path]:
    # Typically at repo root, but search recursively as a fallback
    direct = repo_dir / "stats.xml"
    if direct.exists():
        return direct
    return find_file_recursive(repo_dir, "stats.xml")


def find_split_file(repo_dir: Path, split: str) -> Optional[Path]:
    # Typical naming: <lang>-<treebank>-ud-<split>.conllu somewhere in repo
    suffix = f"-ud-{split}.conllu"
    return find_file_recursive(repo_dir, suffix)


def find_train_files(repo_dir: Path) -> List[Path]:
    """Some treebanks shard TRAIN into multiple files (e.g., *-ud-train-a/b/c.conllu or many parts).
    Return a sorted list of all train files if present; otherwise an empty list.
    """
    candidates: List[Path] = []
    # Look for any file that contains -ud-train and ends with .conllu (but exclude dev/test explicitly)
    # ONLY search in the top-level directory to avoid subdirectory files
    for p in repo_dir.glob("*.conllu"):
        name = p.name
        low = name.lower()
        if "-ud-train" in low and not ("-ud-dev" in low or "-ud-test" in low):
            candidates.append(p)
    candidates.sort()
    return candidates


def find_dev_files(repo_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    # ONLY search in the top-level directory to avoid subdirectory files
    for p in repo_dir.glob("*.conllu"):
        low = p.name.lower()
        if "-ud-dev" in low and "enhanced" not in low:
            candidates.append(p)
    candidates.sort()
    return candidates


def find_test_files(repo_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    # ONLY search in the top-level directory to avoid subdirectory files
    for p in repo_dir.glob("*.conllu"):
        low = p.name.lower()
        if "-ud-test" in low and "enhanced" not in low:
            candidates.append(p)
    candidates.sort()
    return candidates


def parse_ud_stats(stats_xml_path: Path) -> Dict[str, Optional[str]]:
    """Parse UD stats.xml supporting both attribute-style and element-style formats.

    - Some releases use attributes on <train>/<dev>/<test>; others nest counts under <size>.
    - UPOS types may be under <upos unique> or as <tags unique>.
    - Deprel types may be <deprels unique> or <deps unique>.
    - Multiword tokens may be reported as <multiword_tokens count> or <size>/<total>/<fused>.
    """
    root = ET.parse(stats_xml_path).getroot()
    out: Dict[str, Optional[str]] = {}

    # Basic metadata (may be missing in some repos)
    out["ud_release"] = root.attrib.get("ud-release")
    out["treebank_name"] = root.attrib.get("treebank")
    out["language"] = root.attrib.get("language")
    out["license"] = root.attrib.get("license")

    # Helper to coerce text to str numbers
    def _txt(node: Optional[ET.Element], child_name: str) -> Optional[str]:
        if node is None:
            return None
        c = node.find(child_name)
        return c.text.strip() if c is not None and c.text is not None else None

    # Split sizes: try attribute-style; fallback to element-style under <size>
    size_node = root.find(".//size")
    for split in ["train", "dev", "test"]:
        node_attr = root.find(f".//{split}")
        sent = node_attr.attrib.get("sentences") if node_attr is not None else None
        tok = node_attr.attrib.get("tokens") if node_attr is not None else None
        avg = node_attr.attrib.get("tokens_per_sentence") if node_attr is not None else None
        if (sent is None or tok is None) and size_node is not None:
            node_elem = size_node.find(split)
            sent = _txt(node_elem, "sentences") or sent
            tok = _txt(node_elem, "tokens") or tok
            # compute average if possible
            try:
                if sent and tok and (avg is None or avg == ""):
                    s = int(sent)
                    t = int(tok)
                    if s > 0:
                        avg = f"{t / s:.3f}"
            except Exception:
                pass
        out[f"n_{split}_sent"] = sent
        out[f"n_{split}_tokens"] = tok
        out[f"{split}_avg_sent_len"] = avg

    # Unique counts
    def _get_unique(name: str) -> Optional[str]:
        node = root.find(f".//{name}")
        return node.attrib.get("unique") if node is not None else None

    out["n_forms"] = _get_unique("forms")
    out["n_lemmas"] = _get_unique("lemmas")
    # UPOS unique: prefer explicit <upos>, else <tags>
    out["n_upos_types"] = _get_unique("upos") or _get_unique("tags")
    # XPOS often missing in modern stats.xml
    out["n_xpos_types"] = _get_unique("xpos")
    # Deprels unique: prefer <deprels>, else <deps>
    out["n_deprel_types"] = _get_unique("deprels") or _get_unique("deps")

    feats_node = root.find(".//feats")
    if feats_node is not None:
        out["n_feat_instances"] = feats_node.attrib.get("count")
        out["n_unique_feat_bundles"] = feats_node.attrib.get("unique")

    # Multiword tokens / fused tokens
    mwt_node = root.find(".//multiword_tokens")
    if mwt_node is not None and mwt_node.attrib.get("count"):
        out["n_mwt_tokens"] = mwt_node.attrib.get("count")
    else:
        fused_txt = _txt(size_node.find("total") if size_node is not None else None, "fused") if size_node is not None else None
        out["n_mwt_tokens"] = fused_txt

    return out


def slugify_treebank_name(name_from_stats: Optional[str], fallback_repo_dir: Path) -> str:
    if name_from_stats and name_from_stats.strip():
        slug = name_from_stats.strip().replace(" ", "-")
        return slug
    # Fallback to repo directory name
    return fallback_repo_dir.name


def filter_czech_pdtc_sentences(input_file: Path, output_file: Path) -> None:
    """Filter Czech PDTC to keep only original PDT sentences (exclude PCEDT, PDTSC, Faust)"""
    pdt_prefixes = ('ln', 'mf', 'cmpr', 'vesm')  # Original PDT prefixes
    
    with open(input_file, 'r', encoding='utf-8') as inf, \
         open(output_file, 'w', encoding='utf-8') as outf:
        
        current_sentence = []
        include_sentence = False
        
        for line in inf:
            if line.startswith('# sent_id = '):
                # Check if this sentence should be included
                sent_id = line.strip().split('= ', 1)[1]
                include_sentence = any(sent_id.startswith(prefix) for prefix in pdt_prefixes)
                current_sentence = [line]
            elif line.strip() == '':
                # End of sentence
                if include_sentence and current_sentence:
                    outf.writelines(current_sentence)
                    outf.write('\n')
                current_sentence = []
                include_sentence = False
            else:
                # Part of current sentence
                if include_sentence:
                    current_sentence.append(line)


def copy_splits_and_stats(
    repo_dir: Path,
    dest_root: Path,
    treebank_slug: str,
    train_srcs: Optional[List[Path]],
    dev_srcs: Optional[List[Path]],
    test_srcs: Optional[List[Path]],
    stats_src: Optional[Path],
) -> Dict[str, Optional[str]]:
    dest_dir = dest_root / treebank_slug
    safe_mkdir(dest_dir)

    # Create a log to record what files are actually concatenated
    concatenation_log = []

    def _copy_opt(src: Optional[Path], dest_name: str) -> Optional[str]:
        if src and src.exists():
            dest_path = dest_dir / dest_name
            shutil.copy2(src, dest_path)
            return str(dest_path)
        return None

    out_paths: Dict[str, Optional[str]] = {}
    # Handle TRAIN: concatenate shards if multiple
    if train_srcs and len(train_srcs) > 0:
        dest_train = dest_dir / "train.conllu"
        concatenation_log.append(f"TRAIN files ({len(train_srcs)}):")
        for src in train_srcs:
            concatenation_log.append(f"  - {src}")
        
        # Check if this is Czech PDTC and needs filtering
        is_czech_pdtc = treebank_slug == "UD_Czech-PDTC"
        
        if is_czech_pdtc:
            # For Czech PDTC, filter to keep only original PDT data
            temp_concat = dest_dir / "temp_train_full.conllu"
            with open(temp_concat, "wb") as out_f:
                for i, src in enumerate(train_srcs):
                    with open(src, "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f)
                    out_f.write(b"\n")
            # Filter the concatenated file
            filter_czech_pdtc_sentences(temp_concat, dest_train)
            temp_concat.unlink()  # Delete temp file
        else:
            # Standard concatenation for other languages
            with open(dest_train, "wb") as out_f:
                for i, src in enumerate(train_srcs):
                    with open(src, "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f)
                    out_f.write(b"\n")
        
        out_paths["train_conllu_path"] = str(dest_train)
    else:
        out_paths["train_conllu_path"] = None
    # DEV: concatenate if multiple
    if dev_srcs and len(dev_srcs) > 0:
        dest_dev = dest_dir / "dev.conllu"
        concatenation_log.append(f"DEV files ({len(dev_srcs)}):")
        for src in dev_srcs:
            concatenation_log.append(f"  - {src}")
        
        if is_czech_pdtc:
            # For Czech PDTC, filter to keep only original PDT data
            temp_concat = dest_dir / "temp_dev_full.conllu"
            with open(temp_concat, "wb") as out_f:
                for src in dev_srcs:
                    with open(src, "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f)
                    out_f.write(b"\n")
            filter_czech_pdtc_sentences(temp_concat, dest_dev)
            temp_concat.unlink()
        else:
            # Standard concatenation for other languages
            with open(dest_dev, "wb") as out_f:
                for src in dev_srcs:
                    with open(src, "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f)
                    out_f.write(b"\n")
        
        out_paths["dev_conllu_path"] = str(dest_dev)
    else:
        out_paths["dev_conllu_path"] = None
    # TEST: concatenate if multiple
    if test_srcs and len(test_srcs) > 0:
        dest_test = dest_dir / "test.conllu"
        concatenation_log.append(f"TEST files ({len(test_srcs)}):")
        for src in test_srcs:
            concatenation_log.append(f"  - {src}")
        
        if is_czech_pdtc:
            # For Czech PDTC, filter to keep only original PDT data
            temp_concat = dest_dir / "temp_test_full.conllu"
            with open(temp_concat, "wb") as out_f:
                for src in test_srcs:
                    with open(src, "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f)
                    out_f.write(b"\n")
            filter_czech_pdtc_sentences(temp_concat, dest_test)
            temp_concat.unlink()
        else:
            # Standard concatenation for other languages
            with open(dest_test, "wb") as out_f:
                for src in test_srcs:
                    with open(src, "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f)
                    out_f.write(b"\n")
        
        out_paths["test_conllu_path"] = str(dest_test)
    else:
        out_paths["test_conllu_path"] = None
    out_paths["stats_xml_path"] = _copy_opt(stats_src, "stats.xml")

    # Write concatenation log
    log_file = dest_dir / "concatenation_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"File concatenation log for {treebank_slug}\n")
        f.write("=" * 50 + "\n\n")
        for line in concatenation_log:
            f.write(line + "\n")

    return out_paths


def ensure_output_dirs(metadata_out_csv: Path) -> None:
    safe_mkdir(metadata_out_csv.parent)


def write_metadata_csv(rows: List[Dict[str, Optional[str]]], out_csv: Path) -> None:
    ensure_output_dirs(out_csv)
    # Define column order for stability
    columns = [
        "language",
        "treebank_name",
        "treebank_slug",
        "repo_url",
        "repo_dir",
        "requested_ref",
        "ud_release_normalized",
        "commit_hash",
        "tag_or_desc",
        "license",
        "ud_release",
        "enhanced_ud_available",
        "n_train_sent",
        "n_train_tokens",
        "train_avg_sent_len",
        "n_dev_sent",
        "n_dev_tokens",
        "dev_avg_sent_len",
        "n_test_sent",
        "n_test_tokens",
        "test_avg_sent_len",
        "n_forms",
        "n_lemmas",
        "n_upos_types",
        "n_xpos_types",
        "n_deprel_types",
        "n_feat_instances",
        "n_unique_feat_bundles",
        "n_mwt_tokens",
        "train_conllu_path",
        "train_conllu_path_sha256",
        "dev_conllu_path",
        "dev_conllu_path_sha256",
        "test_conllu_path",
        "test_conllu_path_sha256",
        "stats_xml_path",
        "stats_xml_path_sha256",
    ]

    # Normalize rows: map missing keys to empty strings
    norm_rows: List[Dict[str, str]] = []
    for r in rows:
        norm: Dict[str, str] = {}
        for c in columns:
            v = r.get(c)
            norm[c] = "" if v is None else str(v)
        norm_rows.append(norm)

    header_needed = not out_csv.exists()
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if header_needed:
            writer.writeheader()
        for r in norm_rows:
            writer.writerow(r)


def process_repo(
    spec: RepoSpec,
    tmp_root: Path,
    dest_root: Path,
    default_ref: Optional[str] = None,
    strict_ref: bool = False,
) -> Dict[str, Optional[str]]:
    repo_basename = Path(spec.url.rstrip("/").split("/")[-1]).with_suffix("").name
    clone_dir = tmp_root / repo_basename

    chosen_ref = spec.ref or default_ref
    try:
        clone_repo(spec.url, clone_dir, chosen_ref, depth=1)
    except Exception as e:
        if chosen_ref and not strict_ref:
            log.warning(
                f"Checkout for ref '{chosen_ref}' failed; proceeding with repo HEAD. Error: {e}"
            )
            if not clone_dir.exists():
                # attempt plain clone without ref
                clone_repo(spec.url, clone_dir, None, depth=1)
        else:
            raise
    commit_hash, tag = detect_commit_and_tag(clone_dir)

    stats_xml_path = find_stats_xml(clone_dir)
    if not stats_xml_path:
        msg = f"stats.xml not found under {clone_dir}"
        if strict_ref:
            raise FileNotFoundError(msg)
        else:
            log.warning(msg)

    stats = parse_ud_stats(stats_xml_path) if stats_xml_path else {}
    treebank_slug = slugify_treebank_name(stats.get("treebank_name"), clone_dir)
    # PUD avoidance guard
    repo_basename = Path(spec.url.rstrip("/").split("/")[-1]).with_suffix("").name
    if repo_basename.endswith("-PUD") or treebank_slug.endswith("-PUD"):
        log.warning(f"{repo_basename}: PUD treebank detected; consider excluding from training.")
    # Use clean repo basename as treebank_slug for consistency
    treebank_slug = repo_basename

    # Prefer shard-aware discovery for TRAIN/DEV/TEST
    train_files = find_train_files(clone_dir)
    if not train_files:
        single_train = find_split_file(clone_dir, "train")
        train_files = [single_train] if single_train else []
    dev_files = find_dev_files(clone_dir)
    if not dev_files:
        single_dev = find_split_file(clone_dir, "dev")
        dev_files = [single_dev] if single_dev else []
    test_files = find_test_files(clone_dir)
    if not test_files:
        single_test = find_split_file(clone_dir, "test")
        test_files = [single_test] if single_test else []

    # Log LUW (ignored) and enhanced UD availability
    luw = list(clone_dir.rglob("*-ud-*-luw.conllu"))
    if luw:
        log.info(f"{clone_dir.name}: Found LUW files ({len(luw)}) which will be ignored.")

    if not dev_files:
        log.warning(f"{clone_dir.name}: No dev split found; leaving dev_* fields blank")
    if not test_files:
        log.warning(f"{clone_dir.name}: No test split found; leaving test_* fields blank")

    copied_paths = copy_splits_and_stats(
        clone_dir,
        dest_root,
        treebank_slug,
        train_files,
        dev_files,
        test_files,
        stats_xml_path,
    )

    row: Dict[str, Optional[str]] = {}
    # License fallback if missing
    license_val = stats.get("license")
    if not license_val:
        license_val = detect_license(clone_dir)

    row.update({
        "language": spec.language or stats.get("language"),
        "treebank_name": stats.get("treebank_name"),
        "treebank_slug": treebank_slug,
        "repo_url": spec.url,
        "repo_dir": str(clone_dir),
        "requested_ref": chosen_ref,
        "ud_release_normalized": normalize_ud_ref(chosen_ref) if chosen_ref else "",
        "commit_hash": commit_hash,
        "tag_or_desc": tag or spec.ref or default_ref,
        "license": license_val,
        "ud_release": stats.get("ud_release"),
        "enhanced_ud_available": str(detect_enhanced(clone_dir)).lower(),
        "n_train_sent": stats.get("n_train_sent"),
        "n_train_tokens": stats.get("n_train_tokens"),
        "train_avg_sent_len": stats.get("train_avg_sent_len"),
        "n_dev_sent": stats.get("n_dev_sent"),
        "n_dev_tokens": stats.get("n_dev_tokens"),
        "dev_avg_sent_len": stats.get("dev_avg_sent_len"),
        "n_test_sent": stats.get("n_test_sent"),
        "n_test_tokens": stats.get("n_test_tokens"),
        "test_avg_sent_len": stats.get("test_avg_sent_len"),
        "n_forms": stats.get("n_forms"),
        "n_lemmas": stats.get("n_lemmas"),
        "n_upos_types": stats.get("n_upos_types"),
        "n_xpos_types": stats.get("n_xpos_types"),
        "n_deprel_types": stats.get("n_deprel_types"),
        "n_feat_instances": stats.get("n_feat_instances"),
        "n_unique_feat_bundles": stats.get("n_unique_feat_bundles"),
        "n_mwt_tokens": stats.get("n_mwt_tokens"),
    })
    row.update(copied_paths)

    # Checksums
    def sha256(path: Path) -> str:
        import hashlib
        if not path or not path.exists():
            return ""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    for key in ["train_conllu_path", "dev_conllu_path", "test_conllu_path", "stats_xml_path"]:
        p = copied_paths.get(key)
        row[f"{key}_sha256"] = sha256(Path(p)) if p else ""

    # Coerce numeric fields
    def _to_int(val: Optional[str]) -> Optional[int]:
        try:
            return int(val) if val not in (None, "") else None
        except Exception:
            return None
    for k in [
        "n_train_sent","n_train_tokens","n_dev_sent","n_dev_tokens","n_test_sent","n_test_tokens",
        "n_forms","n_lemmas","n_upos_types","n_xpos_types","n_deprel_types",
        "n_feat_instances","n_unique_feat_bundles","n_mwt_tokens",
    ]:
        row[k] = _to_int(row.get(k))
    # Derived diagnostics
    try:
        nf = row.get("n_forms") or 0
        nl = row.get("n_lemmas") or 0
        row["lemma_form_ratio"] = (float(nl) / float(nf)) if nf else None
    except Exception:
        row["lemma_form_ratio"] = None
    return row


def detect_enhanced(repo_dir: Path) -> bool:
    # filename check
    if list(repo_dir.rglob("*-enhanced.conllu")):
        return True
    # header check in early comment lines
    for path in repo_dir.rglob("*-ud-*.conllu"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for _ in range(50):
                    line = f.readline()
                    if not line or not line.startswith("#"):
                        break
                    if "global.columns" in line and "DEPS" in line:
                        return True
        except Exception:
            continue
    return False


def detect_license(repo_dir: Path) -> Optional[str]:
    # Try common license files
    for name in ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"]:
        p = repo_dir / name
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")[:2048].lower()
                for key in ["mit", "apache", "bsd", "gpl", "lgpl", "mpl", "cc by", "cc-by", "creativecommons"]:
                    if key in txt:
                        return key.upper()
                return name
            except Exception:
                return name
    # Grep README for a license mention
    for name in ["README", "README.md", "README.txt"]:
        p = repo_dir / name
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")[:4096].lower()
                if "license" in txt:
                    return "SEE README"
            except Exception:
                continue
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download UD repos, parse stats.xml, and log metadata.")
    parser.add_argument("input_table", type=Path, help="CSV/TSV file with columns: url[,language,treebank,ref]")
    parser.add_argument("--tmp_clone_dir", type=Path, default=Path("data/tmp_ud_clones"), help="Where to clone repos")
    parser.add_argument("--dest_root", type=Path, default=Path("data/ud"), help="Where to place consolidated treebank files")
    parser.add_argument("--metadata_out", type=Path, default=Path("data/lang_stats/ud_metadata.csv"), help="Consolidated CSV output")
    parser.add_argument("--default_ref", type=str, default=None, help="Default UD tag/branch/commit to use when a row has no ref/ud_release (e.g., ud-treebanks-v2.14)")
    parser.add_argument("--strict_ref", action="store_true", help="Fail if checkout of requested/default ref is not possible")
    parser.add_argument("--clean_tmp", action="store_true", help="Remove tmp clone directories after processing")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--parallel", type=int, default=4, help="Number of concurrent repos to process (I/O-bound; 4–8 is usually safe)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite metadata CSV instead of appending")
    args = parser.parse_args(argv)

    logging.basicConfig(stream=sys.stdout, level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    specs = read_input_table(args.input_table)
    if not specs:
        log.error("No rows found in input table.")
        return 1

    safe_mkdir(args.tmp_clone_dir)
    safe_mkdir(args.dest_root)
    ensure_output_dirs(args.metadata_out)

    total = len(specs)
    rows_collected: List[Dict[str, Optional[str]]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        pbar = tqdm(total=total, desc="Repos", unit="repo")
        futures = {
            ex.submit(
                process_repo_return_row,
                i,
                total,
                spec,
                args.tmp_clone_dir,
                args.dest_root,
                args.default_ref,
                args.strict_ref,
            ): i
            for i, spec in enumerate(specs, start=1)
        }
        for fut in as_completed(futures):
            try:
                row = fut.result()
                if row:
                    rows_collected.append(row)
            except Exception as e:
                log.error(f"Parallel task failed: {e}", exc_info=True)
            finally:
                pbar.update(1)
        pbar.close()

    # Write once (support overwrite)
    if args.overwrite and args.metadata_out.exists():
        try:
            args.metadata_out.unlink()
        except Exception:
            pass
    if rows_collected:
        write_metadata_csv(rows_collected, args.metadata_out)

    if args.clean_tmp:
        try:
            shutil.rmtree(args.tmp_clone_dir)
            log.info(f"Removed temporary clone directory: {args.tmp_clone_dir}")
        except Exception as e:
            log.warning(f"Could not remove tmp directory {args.tmp_clone_dir}: {e}")

    log.info(f"Wrote metadata rows to {args.metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


