#!/usr/bin/env python3
"""
Review Wikipedia targets for pretraining exposure proxy.

Maps UD languages to Wikipedia codes and checks for potential issues
before downloading dumps. Follows Wu & Dredze (2019) methodology.
"""

from __future__ import annotations

import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

REPO_ROOT = Path(__file__).parent.parent

# Canonical mapping: UD language → Wikipedia code
UD_TO_WIKI_MAPPING = {
    "UD_Arabic-PADT": "ar",
    "UD_Basque-BDT": "eu", 
    "UD_Bulgarian-BTB": "bg",
    "UD_Chinese-GSD": "zh",
    "UD_Czech-PDTC": "cs",
    "UD_English-EWT": "en",
    "UD_Finnish-TDT": "fi",
    "UD_French-GSD": "fr",
    "UD_German-GSD": "de",
    "UD_Greek-GDT": "el",
    "UD_Hebrew-HTB": "he",
    "UD_Hindi-HDTB": "hi",
    "UD_Hungarian-Szeged": "hu",
    "UD_Indonesian-GSD": "id",
    "UD_Japanese-GSD": "ja",
    "UD_Korean-GSD": "ko",
    "UD_Persian-Seraji": "fa",  # Note: fawiki, not perwiki
    "UD_Polish-PDB": "pl",
    "UD_Russian-SynTagRus": "ru",
    "UD_Spanish-AnCora": "es",
    "UD_Turkish-IMST": "tr",
    "UD_Urdu-UDTB": "ur",
    "UD_Vietnamese-VTB": "vi",
}

# Target snapshot dates (prefer 2018-10-01, fallback to closest 2018)
TARGET_DATES = [
    "20181001",  # Primary target (mBERT release window)
    "20181101", "20180901",  # Adjacent months
    "20181201", "20180801",  # Extended range
    "20181015", "20180915",  # Mid-month alternatives
]

def load_ud_languages() -> List[str]:
    """Load our 23 UD languages from analysis table."""
    analysis_file = REPO_ROOT / "outputs" / "analysis" / "analysis_table_L7.csv"
    
    if not analysis_file.exists():
        raise FileNotFoundError(f"Analysis table not found: {analysis_file}")
    
    df = pd.read_csv(analysis_file)
    languages = sorted(df['language_slug'].unique())
    
    print(f"Loaded {len(languages)} UD languages from analysis table")
    return languages

def build_dump_url(wiki_code: str, date: str) -> str:
    """Build Wikipedia dump URL for given wiki and date."""
    return (f"https://dumps.wikimedia.org/{wiki_code}wiki/{date}/"
            f"{wiki_code}wiki-{date}-pages-articles-multistream.xml.bz2")

def check_url_exists(url: str, timeout: int = 10) -> bool:
    """Check if URL exists (HEAD request)."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException:
        return False

def find_best_dump_date(wiki_code: str) -> Optional[str]:
    """Find the best available dump date for a wiki."""
    print(f"  Checking {wiki_code}wiki snapshots...")
    
    for date in TARGET_DATES:
        url = build_dump_url(wiki_code, date)
        print(f"    {date}: ", end="", flush=True)
        
        if check_url_exists(url):
            print("✓ Found")
            return date
        else:
            print("✗ Missing")
    
    print(f"    ⚠ No 2018 snapshots found for {wiki_code}wiki")
    return None

def review_targets() -> pd.DataFrame:
    """Review all targets and build mapping table."""
    print("Reviewing Wikipedia targets for pretraining exposure...")
    print(f"Target date: 2018-10-01 (or closest 2018 snapshot)")
    print(f"Following Wu & Dredze (2019) methodology")
    print()
    
    # Load UD languages
    ud_languages = load_ud_languages()
    
    # Check mapping completeness
    unmapped = [lang for lang in ud_languages if lang not in UD_TO_WIKI_MAPPING]
    if unmapped:
        raise ValueError(f"Unmapped UD languages: {unmapped}")
    
    print(f"✓ All {len(ud_languages)} UD languages have wiki mappings")
    print()
    
    # Build target review table
    targets = []
    
    for lang_slug in ud_languages:
        wiki_code = UD_TO_WIKI_MAPPING[lang_slug]
        
        print(f"📍 {lang_slug} → {wiki_code}wiki")
        
        # Find best dump date
        best_date = find_best_dump_date(wiki_code)
        
        if best_date:
            dump_url = build_dump_url(wiki_code, best_date)
            status = "✓ Ready"
        else:
            dump_url = "NOT_FOUND"
            status = "✗ Missing"
        
        targets.append({
            'language_slug': lang_slug,
            'wiki_code': wiki_code,
            'target_date': best_date,
            'dump_url': dump_url,
            'status': status,
        })
        
        print(f"    → {status}")
        print()
    
    return pd.DataFrame(targets)

def analyze_targets(targets_df: pd.DataFrame) -> None:
    """Analyze target feasibility and potential issues."""
    print("=" * 60)
    print("TARGET ANALYSIS")
    print("=" * 60)
    
    # Success rate
    found = targets_df['status'].str.contains('Ready').sum()
    total = len(targets_df)
    print(f"Success rate: {found}/{total} ({found/total*100:.1f}%)")
    print()
    
    # Missing targets
    missing = targets_df[targets_df['status'].str.contains('Missing')]
    if len(missing) > 0:
        print("Missing dumps:")
        for _, row in missing.iterrows():
            print(f"  ✗ {row['language_slug']} ({row['wiki_code']}wiki)")
        print()
    
    # Date distribution
    date_counts = targets_df['target_date'].value_counts().sort_index()
    print("Date distribution:")
    for date, count in date_counts.items():
        if pd.notna(date):
            print(f"  {date}: {count} languages")
    print()
    
    # Special cases
    special_cases = []
    for _, row in targets_df.iterrows():
        lang = row['language_slug']
        wiki = row['wiki_code']
        
        # Flag notable mappings
        if wiki == 'fa':
            special_cases.append(f"{lang} → fawiki (Persian, not perwiki)")
        elif wiki == 'zh':
            special_cases.append(f"{lang} → zhwiki (aggregates script variants)")
        elif wiki == 'he':
            special_cases.append(f"{lang} → hewiki (Hebrew)")
    
    if special_cases:
        print("Special mappings to verify:")
        for case in special_cases:
            print(f"  📝 {case}")
        print()
    
    # Expected size range
    print("Expected WikiSize range: ~10-13 log₂(MB)")
    print("  (tiny wikis ~10, large wikis ~13)")
    print()
    
    # Methodology confirmation
    print("Methodology confirmation:")
    print("  ✓ Using compressed .bz2 archive size (Wu & Dredze 2019)")
    print("  ✓ Target: 2018 snapshots (mBERT training window)")
    print("  ✓ Formula: WikiSize = log₂(compressed_MB)")
    print("  ✓ All 23 UD languages mapped to canonical wiki codes")

def main():
    """Main entry point."""
    print("Stage 4B: Pretraining Exposure Proxy")
    print("Reviewing Wikipedia dump targets...")
    print()
    
    # Review targets
    targets_df = review_targets()
    
    # Save target review
    review_file = REPO_ROOT / "outputs" / "analysis" / "wiki_targets_review.csv"
    targets_df.to_csv(review_file, index=False)
    print(f"💾 Saved target review: {review_file}")
    print()
    
    # Analyze feasibility
    analyze_targets(targets_df)
    
    print("=" * 60)
    print("READY FOR DOWNLOAD PHASE")
    print("=" * 60)
    print("Next: Run download script to fetch dumps and compute WikiSize")

if __name__ == "__main__":
    main()
