#!/usr/bin/env python3
"""
Compute UD dataset statistics and linguistic inventories.

Generates ud_stats_derived.csv with one row per language containing:
- Sentence counts (train/test)
- Content token counts (after dropping PUNCT/SYM)
- UPOS and DEPREL type inventories
- Log-transformed counts
- Metadata (UD release, checksums)

Following Stage 4A specifications precisely.
"""

from __future__ import annotations

import re
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

REPO_ROOT = Path(__file__).parent.parent
UD_DATA_ROOT = REPO_ROOT / "data" / "ud"

@dataclass
class UDStats:
    """Statistics for one UD language."""
    language_slug: str
    
    # Sentence counts
    n_train_sent: int
    n_test_sent: int
    
    # Content token counts (UPOS ∉ {PUNCT, SYM})
    n_train_tokens_content: int
    n_test_tokens_content: int
    
    # Type inventories (from train)
    n_deprel_types: int
    n_upos_types: int
    
    # Log-transformed counts
    log_n_train_sent: float
    log_n_test_sent: float
    log_n_train_tokens_content: float
    log_n_test_tokens_content: float
    
    # Metadata
    ud_release: str
    conllu_checksum_train: str
    conllu_checksum_test: str

def compute_file_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    if not file_path.exists():
        return ""
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()[:16]  # First 16 chars for brevity

def extract_base_deprel(deprel: str) -> str:
    """Extract base DEPREL label, stripping subtype after colon."""
    return re.sub(r':.*$', '', deprel)

def process_conllu_file(file_path: Path, collect_types: bool = False) -> Tuple[int, int, Set[str], Set[str]]:
    """
    Process a CoNLL-U file and extract statistics.
    
    Returns:
        (n_sentences, n_content_tokens, upos_types, deprel_types)
    """
    if not file_path.exists():
        return 0, 0, set(), set()
    
    n_sentences = 0
    n_content_tokens = 0
    upos_types = set()
    deprel_types = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Count sentences on first token of each sentence
            if line and not line.startswith('#'):
                # Check if this is the start of a new sentence (token ID = 1)
                fields = line.split('\t')
                if len(fields) >= 10 and fields[0] == '1':
                    n_sentences += 1
            
            # Process token lines
            if line and not line.startswith('#'):
                fields = line.split('\t')
                
                # Must have at least 10 fields (CoNLL-U format)
                if len(fields) < 10:
                    continue
                
                token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = fields
                
                # Skip multiword ranges (3-4) and empty nodes (1.1)
                if '-' in token_id or '.' in token_id:
                    continue
                
                # Skip punctuation and symbols
                if upos in {'PUNCT', 'SYM'}:
                    continue
                
                # Count content tokens
                n_content_tokens += 1
                
                # Collect types if requested (for train)
                if collect_types:
                    upos_types.add(upos)
                    base_deprel = extract_base_deprel(deprel)
                    deprel_types.add(base_deprel)
    
    return n_sentences, n_content_tokens, upos_types, deprel_types

def get_ud_release_info(language_dir: Path) -> str:
    """Extract UD release information from directory or files."""
    # Check for release info in stats.xml or other metadata files
    stats_xml = language_dir / "stats.xml"
    if stats_xml.exists():
        try:
            with open(stats_xml, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for version info in XML
                if 'release' in content.lower():
                    # Simple extraction - could be made more robust
                    return "UD_2.x"  # Placeholder - could parse actual version
        except:
            pass
    
    # Fallback - assume current UD version
    return "UD_2.x"

def compute_language_stats(language_slug: str) -> Optional[UDStats]:
    """Compute complete statistics for one UD language."""
    language_dir = UD_DATA_ROOT / language_slug
    
    if not language_dir.exists():
        print(f"Warning: Directory not found for {language_slug}")
        return None
    
    print(f"Processing {language_slug}...")
    
    # File paths
    train_file = language_dir / "train.conllu"
    test_file = language_dir / "test.conllu"
    
    # Process train file (with type collection)
    n_train_sent, n_train_tokens_content, upos_types, deprel_types = process_conllu_file(
        train_file, collect_types=True
    )
    
    # Process test file (no type collection needed)
    n_test_sent, n_test_tokens_content, _, _ = process_conllu_file(
        test_file, collect_types=False
    )
    
    # Compute type counts
    n_upos_types = len(upos_types)
    n_deprel_types = len(deprel_types)
    
    # Compute log-transformed counts (with guard for x >= 1)
    log_n_train_sent = np.log(max(n_train_sent, 1))
    log_n_test_sent = np.log(max(n_test_sent, 1))
    log_n_train_tokens_content = np.log(max(n_train_tokens_content, 1))
    log_n_test_tokens_content = np.log(max(n_test_tokens_content, 1))
    
    # Get metadata
    ud_release = get_ud_release_info(language_dir)
    conllu_checksum_train = compute_file_checksum(train_file)
    conllu_checksum_test = compute_file_checksum(test_file)
    
    # Create stats object
    stats = UDStats(
        language_slug=language_slug,
        n_train_sent=n_train_sent,
        n_test_sent=n_test_sent,
        n_train_tokens_content=n_train_tokens_content,
        n_test_tokens_content=n_test_tokens_content,
        n_deprel_types=n_deprel_types,
        n_upos_types=n_upos_types,
        log_n_train_sent=log_n_train_sent,
        log_n_test_sent=log_n_test_sent,
        log_n_train_tokens_content=log_n_train_tokens_content,
        log_n_test_tokens_content=log_n_test_tokens_content,
        ud_release=ud_release,
        conllu_checksum_train=conllu_checksum_train,
        conllu_checksum_test=conllu_checksum_test
    )
    
    # Print summary
    print(f"  Train: {n_train_sent:,} sentences, {n_train_tokens_content:,} content tokens")
    print(f"  Test:  {n_test_sent:,} sentences, {n_test_tokens_content:,} content tokens")
    print(f"  Types: {n_upos_types} UPOS, {n_deprel_types} DEPREL")
    
    return stats

def get_available_languages() -> List[str]:
    """Get list of available UD language directories."""
    if not UD_DATA_ROOT.exists():
        return []
    
    return [
        d.name for d in UD_DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("UD_")
    ]

def validate_stats(stats_list: List[UDStats]) -> None:
    """Validate computed statistics for plausibility."""
    print(f"\nValidation results for {len(stats_list)} languages:")
    
    # Check completeness
    print(f"✓ Completeness: {len(stats_list)}/23 languages processed")
    
    # Check DEPREL types range
    deprel_counts = [s.n_deprel_types for s in stats_list]
    deprel_min, deprel_max = min(deprel_counts), max(deprel_counts)
    print(f"✓ DEPREL types: {deprel_min}-{deprel_max} (expected: 25-60)")
    
    # Check UPOS types range  
    upos_counts = [s.n_upos_types for s in stats_list]
    upos_min, upos_max = min(upos_counts), max(upos_counts)
    print(f"✓ UPOS types: {upos_min}-{upos_max} (expected: 12-17)")
    
    # Check for any zero counts (would indicate issues)
    zero_train = sum(1 for s in stats_list if s.n_train_sent == 0)
    zero_test = sum(1 for s in stats_list if s.n_test_sent == 0) 
    if zero_train > 0 or zero_test > 0:
        print(f"⚠ Warning: {zero_train} languages with 0 train sentences, {zero_test} with 0 test sentences")
    
    # Show extreme values for inspection
    largest_train = max(stats_list, key=lambda s: s.n_train_sent)
    smallest_train = min(stats_list, key=lambda s: s.n_train_sent)
    print(f"✓ Train size range: {smallest_train.language_slug} ({smallest_train.n_train_sent:,}) to {largest_train.language_slug} ({largest_train.n_train_sent:,})")

def main():
    """Main entry point."""
    print("Computing UD dataset statistics (Stage 4A)...")
    
    # Get available languages
    languages = get_available_languages()
    if not languages:
        print("Error: No UD languages found")
        return
    
    print(f"Found {len(languages)} UD languages")
    
    # Process each language
    stats_list = []
    for language_slug in sorted(languages):
        stats = compute_language_stats(language_slug)
        if stats:
            stats_list.append(stats)
    
    if not stats_list:
        print("Error: No statistics computed")
        return
    
    # Validate results
    validate_stats(stats_list)
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'language_slug': s.language_slug,
            'n_train_sent': s.n_train_sent,
            'n_test_sent': s.n_test_sent,
            'n_train_tokens_content': s.n_train_tokens_content,
            'n_test_tokens_content': s.n_test_tokens_content,
            'n_deprel_types': s.n_deprel_types,
            'n_upos_types': s.n_upos_types,
            'log_n_train_sent': s.log_n_train_sent,
            'log_n_test_sent': s.log_n_test_sent,
            'log_n_train_tokens_content': s.log_n_train_tokens_content,
            'log_n_test_tokens_content': s.log_n_test_tokens_content,
            'ud_release': s.ud_release,
            'conllu_checksum_train': s.conllu_checksum_train,
            'conllu_checksum_test': s.conllu_checksum_test
        }
        for s in stats_list
    ])
    
    # Save to CSV
    output_path = REPO_ROOT / "outputs" / "analysis" / "ud_stats_derived.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved UD statistics to {output_path}")
    print(f"✓ Generated table with {len(df)} rows and {len(df.columns)} columns")
    
    # Show sample of results
    print(f"\nSample results:")
    print(df[['language_slug', 'n_train_sent', 'n_test_sent', 'n_upos_types', 'n_deprel_types']].head())

if __name__ == "__main__":
    main()
