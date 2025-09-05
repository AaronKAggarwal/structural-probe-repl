#!/usr/bin/env python3
"""
Investigate FEATS coverage for low-complexity languages.

Analyze Vietnamese, Japanese, and Korean to understand why they have
near-zero morphological complexity scores.
"""

from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

UD_DATA_ROOT = Path(__file__).parent.parent / "data" / "ud"

def analyze_feats_coverage(language_slug: str) -> Dict:
    """Analyze FEATS coverage and bundle inventory for one language."""
    print(f"\n{'='*60}")
    print(f"FEATS ANALYSIS: {language_slug}")
    print('='*60)
    
    # Find train file
    language_dir = UD_DATA_ROOT / language_slug
    train_file = language_dir / "train.conllu"
    
    if not train_file.exists():
        print(f"❌ Train file not found: {train_file}")
        return {}
    
    # Parse and analyze
    content_tokens = []
    all_sentences = []
    current_sentence = []
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not line:
                # End of sentence
                if current_sentence:
                    all_sentences.append(current_sentence.copy())
                    current_sentence = []
            elif line.startswith('#'):
                # Skip comments
                continue
            else:
                # Parse token line
                fields = line.split('\t')
                if len(fields) >= 10:
                    token_id = fields[0]
                    form = fields[1]
                    upos = fields[3]
                    feats = fields[5]
                    
                    # Skip MWT ranges and empty nodes
                    if '-' in token_id or '.' in token_id:
                        continue
                    
                    token_data = {
                        'line_num': line_num,
                        'id': token_id,
                        'form': form,
                        'upos': upos,
                        'feats': feats
                    }
                    
                    current_sentence.append(token_data)
                    
                    # Collect content tokens (exclude PUNCT, SYM)
                    if upos not in {'PUNCT', 'SYM'}:
                        content_tokens.append(token_data)
    
    # Handle final sentence
    if current_sentence:
        all_sentences.append(current_sentence)
    
    print(f"📊 BASIC STATS")
    print(f"   Total sentences: {len(all_sentences):,}")
    print(f"   Total content tokens: {len(content_tokens):,}")
    
    # FEATS coverage analysis
    non_empty_feats = [token for token in content_tokens if token['feats'] != '_']
    coverage_rate = len(non_empty_feats) / len(content_tokens) if content_tokens else 0
    
    print(f"\n📈 FEATS COVERAGE")
    print(f"   Content tokens with FEATS ≠ '_': {len(non_empty_feats):,}/{len(content_tokens):,}")
    print(f"   Coverage rate: {coverage_rate:.1%}")
    
    # Bundle inventory
    feats_counter = Counter(token['feats'] for token in content_tokens)
    
    print(f"\n📋 BUNDLE INVENTORY (Top 10)")
    for i, (bundle, count) in enumerate(feats_counter.most_common(10), 1):
        percentage = count / len(content_tokens) * 100
        display_bundle = bundle if bundle != '_' else "_ (empty)"
        print(f"   {i:2d}. {display_bundle:30} {count:8,} ({percentage:5.1f}%)")
    
    # Sample sentences spot-check
    print(f"\n🔍 SPOT-CHECK: Sample sentences")
    sample_sentences = all_sentences[:5]  # First 5 sentences
    
    for sent_idx, sentence in enumerate(sample_sentences, 1):
        print(f"\n   Sentence {sent_idx}:")
        content_tokens_in_sent = [token for token in sentence if token['upos'] not in {'PUNCT', 'SYM'}]
        
        for token in content_tokens_in_sent[:8]:  # Show up to 8 content tokens
            feats_display = token['feats'] if token['feats'] != '_' else '(empty)'
            print(f"     {token['form']:15} {token['upos']:8} → FEATS: {feats_display}")
        
        if len(content_tokens_in_sent) > 8:
            print(f"     ... ({len(content_tokens_in_sent) - 8} more tokens)")
    
    # Unique FEATS analysis for non-empty
    if non_empty_feats:
        print(f"\n🎯 NON-EMPTY FEATS ANALYSIS")
        unique_non_empty = set(token['feats'] for token in non_empty_feats)
        print(f"   Unique non-empty bundles: {len(unique_non_empty)}")
        
        # Show sample non-empty FEATS
        sample_non_empty = list(unique_non_empty)[:10]
        print(f"   Sample non-empty bundles:")
        for bundle in sample_non_empty:
            print(f"     {bundle}")
    
    return {
        'language_slug': language_slug,
        'n_sentences': len(all_sentences),
        'n_content_tokens': len(content_tokens),
        'n_non_empty_feats': len(non_empty_feats),
        'coverage_rate': coverage_rate,
        'top_bundle': feats_counter.most_common(1)[0] if feats_counter else None,
        'unique_bundles': len(feats_counter)
    }

def main():
    """Investigate the three low-complexity languages."""
    print("FEATS COVERAGE INVESTIGATION")
    print("Analyzing Vietnamese, Japanese, and Korean")
    
    target_languages = [
        'UD_Vietnamese-VTB',
        'UD_Japanese-GSD', 
        'UD_Korean-GSD'
    ]
    
    results = []
    
    for language_slug in target_languages:
        result = analyze_feats_coverage(language_slug)
        if result:
            results.append(result)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print('='*60)
    
    print(f"{'Language':<20} {'Tokens':<10} {'Coverage':<10} {'Top Bundle':<15} {'Share':<8}")
    print("-" * 70)
    
    for result in results:
        lang_short = result['language_slug'].replace('UD_', '').replace('-GSD', '').replace('-VTB', '')
        tokens = f"{result['n_content_tokens']:,}"
        coverage = f"{result['coverage_rate']:.1%}"
        
        if result['top_bundle']:
            top_bundle, top_count = result['top_bundle']
            top_bundle_display = "empty" if top_bundle == '_' else top_bundle[:12]
            top_share = f"{top_count/result['n_content_tokens']*100:.1f}%"
        else:
            top_bundle_display = "N/A"
            top_share = "N/A"
        
        print(f"{lang_short:<20} {tokens:<10} {coverage:<10} {top_bundle_display:<15} {top_share:<8}")
    
    print(f"\n🔍 INTERPRETATION:")
    print("   • Coverage ~0%: Language has virtually no FEATS annotation")
    print("   • Top bundle ~100% empty: Explains zero entropy (no variation)")
    print("   • This reflects UD annotation policies, not linguistic morphology")

if __name__ == "__main__":
    main()
