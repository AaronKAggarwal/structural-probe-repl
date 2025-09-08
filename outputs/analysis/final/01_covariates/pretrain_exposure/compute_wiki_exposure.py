#!/usr/bin/env python3
"""
Compute pretraining exposure proxy using Wikipedia dump sizes.

Follows Wu & Dredze (2019) methodology with systematic fallback:
1. Check dumps.wikimedia.org/{code}wiki/{date}/dumpstatus.json
2. If 404, try nearby 2018 months
3. If all 404, query Internet Archive metadata
4. Compute WikiSize = log₂(size_MB)
"""

from __future__ import annotations

import math
import requests
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent

# Target dates in priority order (prefer October 2018)
TARGET_DATES = [
    "20181001", "20181101", "20180901",
    "20181201", "20180801", "20181015",
    "20180915", "20180715", "20180615"
]

# UD to Wikipedia mapping
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
    "UD_Persian-Seraji": "fa",
    "UD_Polish-PDB": "pl",
    "UD_Russian-SynTagRus": "ru",
    "UD_Spanish-AnCora": "es",
    "UD_Turkish-IMST": "tr",
    "UD_Urdu-UDTB": "ur",
    "UD_Vietnamese-VTB": "vi",
}

@dataclass
class WikiExposure:
    """Wikipedia exposure data for one language."""
    language_slug: str
    wiki_code: str
    chosen_date: str
    source: str  # 'wikimedia' or 'archive_org'
    size_bytes: int
    size_mb: float
    wiki_size_log2_mb: float
    dump_url: str
    status: str

def fetch_json(url: str, timeout: int = 10) -> Optional[Dict]:
    """Fetch JSON from URL, return None if failed."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def get_multistream_size_from_status(status_data: Dict) -> Optional[int]:
    """Extract multistream file size from dumpstatus.json."""
    try:
        jobs = status_data.get('jobs', {})

        # Look for articlesmultistreamdump job
        for job_name, job_data in jobs.items():
            if 'articlesmultistream' in job_name.lower():
                files = job_data.get('files', {})

                # Find the main multistream file
                for filename, file_data in files.items():
                    if filename.endswith('pages-articles-multistream.xml.bz2'):
                        size = file_data.get('size')
                        if size and isinstance(size, (int, str)):
                            return int(size)
        return None
    except Exception:
        return None

def try_wikimedia_dumps(wiki_code: str) -> Optional[Tuple[str, int, str]]:
    """Try to get dump size from dumps.wikimedia.org."""
    print(f"  Checking Wikimedia dumps for {wiki_code}wiki...")
    for date in TARGET_DATES:
        status_url = f"https://dumps.wikimedia.org/{wiki_code}wiki/{date}/dumpstatus.json"
        print(f"    {date}: ", end="", flush=True)
        status_data = fetch_json(status_url)
        if status_data:
            size_bytes = get_multistream_size_from_status(status_data)
            if size_bytes:
                dump_url = f"https://dumps.wikimedia.org/{wiki_code}wiki/{date}/{wiki_code}wiki-{date}-pages-articles-multistream.xml.bz2"
                print(f"Found ({size_bytes:,} bytes)")
                return date, size_bytes, dump_url
            else:
                print("Status found, but no multistream file")
        else:
            print("No status")
        # Small delay to be respectful
        time.sleep(0.5)
    print("    No 2018 dumps found on Wikimedia")
    return None

def try_archive_org(wiki_code: str) -> Optional[Tuple[str, int, str]]:
    """Try to get dump size from Internet Archive."""
    print(f"  Checking Internet Archive for {wiki_code}wiki...")
    for date in TARGET_DATES:
        # Try different IA identifier patterns
        identifiers = [
            f"{wiki_code}wiki-{date}",
            f"wikipedia-{wiki_code}-{date}",
            f"{wiki_code}-wikipedia-{date}"
        ]
        for identifier in identifiers:
            metadata_url = f"https://archive.org/metadata/{identifier}"
            print(f"    {date} ({identifier}): ", end="", flush=True)
            metadata = fetch_json(metadata_url)
            if metadata and 'files' in metadata:
                # Look for multistream file
                for file_info in metadata['files']:
                    filename = file_info.get('name', '')
                    if filename.endswith('pages-articles-multistream.xml.bz2'):
                        size_str = file_info.get('size', '0')
                        size_bytes = int(size_str) if size_str.isdigit() else 0
                        if size_bytes > 0:
                            archive_url = f"https://archive.org/download/{identifier}/{filename}"
                            print(f"Found ({size_bytes:,} bytes)")
                            return date, size_bytes, archive_url
                print("Metadata found, but no multistream file")
            else:
                print("No metadata")
            time.sleep(0.5)
    print("    No archives found on Internet Archive")
    return None

def compute_wiki_exposure(language_slug: str) -> Optional[WikiExposure]:
    """Compute Wikipedia exposure for one language."""
    wiki_code = UD_TO_WIKI_MAPPING[language_slug]
    print(f"Processing {language_slug} -> {wiki_code}wiki")
    # Try Wikimedia first
    wikimedia_result = try_wikimedia_dumps(wiki_code)
    if wikimedia_result:
        date, size_bytes, dump_url = wikimedia_result
        source = "wikimedia"
        status = "Wikimedia dump found"
    else:
        # Fallback to Internet Archive
        archive_result = try_archive_org(wiki_code)
        if archive_result:
            date, size_bytes, dump_url = archive_result
            source = "archive_org"
            status = "Archive.org dump found"
        else:
            print(f"No dumps found for {wiki_code}wiki")
            return None
    # Compute WikiSize
    size_mb = size_bytes / (1024 * 1024)  # Convert to MB
    wiki_size_log2_mb = math.log2(size_mb) if size_mb > 0 else 0.0
    print(f"    Result: {status}: {size_mb:.1f} MB, WikiSize = {wiki_size_log2_mb:.3f}")
    print()
    return WikiExposure(
        language_slug=language_slug,
        wiki_code=wiki_code,
        chosen_date=date,
        source=source,
        size_bytes=size_bytes,
        size_mb=size_mb,
        wiki_size_log2_mb=wiki_size_log2_mb,
        dump_url=dump_url,
        status=status
    )

def load_ud_languages() -> List[str]:
    """Load UD languages from analysis table."""
    analysis_file = REPO_ROOT / "outputs" / "analysis" / "analysis_table_L7.csv"
    df = pd.read_csv(analysis_file)
    return sorted(df['language_slug'].unique())

def main():
    """Main entry point."""
    print("Stage 4B: Computing Wikipedia Exposure Proxy")
    print("Following Wu & Dredze (2019) methodology")
    print("Target: 2018 Wikipedia dumps (mBERT training window)")
    print()
    # Load languages
    languages = load_ud_languages()
    print(f"Processing {len(languages)} UD languages...")
    print()
    # Compute exposure for each language
    results = []
    success_count = 0
    for language_slug in languages:
        exposure = compute_wiki_exposure(language_slug)
        if exposure:
            results.append(exposure)
            success_count += 1
    # Summary
    print("-" * 60)
    print("WIKI EXPOSURE RESULTS")
    print("-" * 60)
    print(f"Success rate: {success_count}/{len(languages)} ({success_count/len(languages)*100:.1f}%)")
    print()
    if results:
        # Convert to DataFrame
        df_data = []
        for exp in results:
            df_data.append({
                'language_slug': exp.language_slug,
                'wiki_code': exp.wiki_code,
                'chosen_date': exp.chosen_date,
                'source': exp.source,
                'size_bytes': exp.size_bytes,
                'size_mb': exp.size_mb,
                'wiki_size_log2_mb': exp.wiki_size_log2_mb,
                'dump_url': exp.dump_url,
                'status': exp.status
            })
        df = pd.DataFrame(df_data)
        # Save results
        output_file = REPO_ROOT / "outputs" / "analysis" / "final" / "01_covariates" / "pretrain_exposure" / "pretrain_exposure.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved exposure data: {output_file}")
        # Statistics
        print("\nWikiSize statistics:")
        print(f"  Range: {df['wiki_size_log2_mb'].min():.3f} - {df['wiki_size_log2_mb'].max():.3f}")
        print(f"  Mean: {df['wiki_size_log2_mb'].mean():.3f}")
        print(f"  Median: {df['wiki_size_log2_mb'].median():.3f}")
        # Source breakdown
        source_counts = df['source'].value_counts()
        print("\nSource breakdown:")
        for source, count in source_counts.items():
            print(f"  {source}: {count} languages")
        # Show sample results
        print("\nSample results:")
        sample_cols = ['language_slug', 'wiki_code', 'wiki_size_log2_mb', 'source']
        print(df[sample_cols].head().to_string(index=False))
        print("Ready to merge into analysis tables")
    else:
        print("No exposure data found for any language")

if __name__ == "__main__":
    main()
