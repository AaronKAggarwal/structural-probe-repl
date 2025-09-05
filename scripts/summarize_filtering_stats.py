#!/usr/bin/env python3
"""
Extract and summarize sentence filtering statistics from extraction logs.
"""

import argparse
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def extract_filtering_stats_from_log(log_path: Path) -> Dict[str, any]:
    """Extract filtering statistics from a single extraction log."""
    stats = {
        'log_path': str(log_path),
        'treebank_slug': 'unknown',
        'train_total': 0, 'train_saved': 0, 'train_skipped': 0,
        'dev_total': 0, 'dev_saved': 0, 'dev_skipped': 0,
        'test_total': 0, 'test_saved': 0, 'test_skipped': 0,
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract treebank slug from file paths in the log
        path_match = re.search(r"/data/ud/([^/]+)/", content)
        if path_match:
            stats['treebank_slug'] = path_match.group(1)
        
        # Pattern for extraction completion lines with optional skipped sentences
        # Example: "Finished processing 'conllu_train'. Saved 6074/6075 sentences to ... [Skipped X sentences due to length > 512 tokens.]"
        completion_pattern = r"Finished processing 'conllu_(\w+)'\. Saved (\d+)/(\d+) sentences.*?(?:Skipped (\d+) sentences due to length|(?=\[|$))"
        
        matches = re.findall(completion_pattern, content, re.DOTALL)
        
        for match in matches:
            split_name = match[0]  # train, dev, test
            saved = int(match[1])
            total = int(match[2])
            skipped = int(match[3]) if match[3] else 0
            
            if split_name in ['train', 'dev', 'test']:
                stats[f'{split_name}_total'] = total
                stats[f'{split_name}_saved'] = saved
                stats[f'{split_name}_skipped'] = skipped
    
    except Exception as e:
        log.warning(f"Error processing {log_path}: {e}")
        stats['error'] = str(e)
    
    return stats


def find_extraction_logs(base_dir: Path) -> List[Path]:
    """Find all extraction log files."""
    outputs_dir = base_dir / "outputs"
    if not outputs_dir.exists():
        return []
    
    log_files = list(outputs_dir.glob("**/extract_embeddings.log"))
    log.info(f"Found {len(log_files)} extraction log files")
    return log_files


def summarize_filtering_stats(base_dir: Path, output_csv: Path) -> None:
    """Generate a summary CSV of filtering statistics."""
    log_files = find_extraction_logs(base_dir)
    
    if not log_files:
        log.error("No extraction log files found")
        return
    
    all_stats = []
    
    for log_path in log_files:
        stats = extract_filtering_stats_from_log(log_path)
        all_stats.append(stats)
    
    # Group by treebank (in case there are multiple runs)
    treebank_stats = {}
    for stats in all_stats:
        slug = stats['treebank_slug']
        if slug == 'unknown':
            continue
            
        if slug not in treebank_stats:
            treebank_stats[slug] = stats
        else:
            # Keep the most recent (assume later log files are newer)
            if stats['log_path'] > treebank_stats[slug]['log_path']:
                treebank_stats[slug] = stats
    
    # Write summary CSV
    fieldnames = [
        'treebank_slug',
        'train_total', 'train_saved', 'train_skipped', 'train_filter_rate',
        'dev_total', 'dev_saved', 'dev_skipped', 'dev_filter_rate', 
        'test_total', 'test_saved', 'test_skipped', 'test_filter_rate',
        'total_sentences', 'total_saved', 'total_skipped', 'overall_filter_rate',
        'log_path'
    ]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for slug, stats in sorted(treebank_stats.items()):
            # Calculate filter rates
            for split in ['train', 'dev', 'test']:
                total = stats[f'{split}_total']
                skipped = stats[f'{split}_skipped']
                stats[f'{split}_filter_rate'] = f"{skipped/total*100:.2f}%" if total > 0 else "0.00%"
            
            # Overall statistics
            total_sentences = stats['train_total'] + stats['dev_total'] + stats['test_total']
            total_saved = stats['train_saved'] + stats['dev_saved'] + stats['test_saved']
            total_skipped = stats['train_skipped'] + stats['dev_skipped'] + stats['test_skipped']
            
            stats.update({
                'total_sentences': total_sentences,
                'total_saved': total_saved, 
                'total_skipped': total_skipped,
                'overall_filter_rate': f"{total_skipped/total_sentences*100:.2f}%" if total_sentences > 0 else "0.00%"
            })
            
            writer.writerow(stats)
    
    log.info(f"Filtering summary written to: {output_csv}")
    
    # Print summary
    print(f"\n=== SENTENCE FILTERING SUMMARY ===")
    print(f"Analyzed {len(treebank_stats)} treebanks")
    
    if treebank_stats:
        total_sentences_all = sum(s['total_sentences'] for s in treebank_stats.values())
        total_skipped_all = sum(s['total_skipped'] for s in treebank_stats.values())
        overall_rate = total_skipped_all / total_sentences_all * 100 if total_sentences_all > 0 else 0
        
        print(f"Total sentences: {total_sentences_all:,}")
        print(f"Total skipped: {total_skipped_all:,}")
        print(f"Overall filter rate: {overall_rate:.2f}%")
        
        # Show top filtered languages
        high_filter = [(slug, s['total_skipped'], s['overall_filter_rate']) 
                      for slug, s in treebank_stats.items() if s['total_skipped'] > 0]
        high_filter.sort(key=lambda x: x[1], reverse=True)
        
        if high_filter:
            print(f"\nTop filtered languages:")
            for slug, skipped, rate in high_filter[:10]:
                print(f"  {slug}: {skipped} sentences ({rate})")


def main():
    parser = argparse.ArgumentParser(description="Summarize sentence filtering statistics")
    parser.add_argument("--base_dir", type=Path, default=".", help="Base project directory")
    parser.add_argument("--output_csv", type=Path, default="sentence_filtering_summary.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    summarize_filtering_stats(args.base_dir, args.output_csv)


if __name__ == "__main__":
    main()
