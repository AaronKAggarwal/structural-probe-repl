#!/usr/bin/env python3
"""
Compute canonical sentence-level covariates from UD CoNLL-U files.

Policy:
- UPOS-based content filtering: exclude PUNCT/SYM tokens
- Content-only arc length with punctuation head collapse
- Compute once per language/split, cache results for reuse across runs

Outputs:
- outputs/analysis/final/01_covariates/sentence_stats/UD_XX/{dev,test}_content_stats.jsonl
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from torch_probe.utils.conllu_reader import read_conll_file

@dataclass
class SentenceStats:
    """Statistics for a single sentence."""
    sent_id: str
    content_len: int  # |I| where I = content tokens (non-PUNCT/SYM)
    mean_arc_len: Optional[float]  # mean |rank_I(h) - rank_I(d)| on content tree
    tree_height: Optional[int]  # max depth from root to any leaf in content tree
    num_content_arcs_used: int  # number of arcs that contributed to mean_arc_len
    orig_len_incl_punct: int  # original sentence length including punctuation
    content_ratio: float  # |I| / orig_len
    uuas_evaluable: bool  # |I| >= 2
    dist_spear_evaluable: bool  # |I| >= 3
    depth_spear_evaluable: bool  # |I| >= 2

def compute_content_indices(upos_tags: List[str]) -> List[int]:
    """Return indices of content tokens (excluding PUNCT/SYM)."""
    return [i for i, tag in enumerate(upos_tags) if tag not in {"PUNCT", "SYM"}]

def content_head_with_collapse(token_idx: int, heads: List[int], upos_tags: List[str]) -> int:
    """
    Find content head for token_idx, collapsing punctuation heads.
    Returns -1 for root, or index of content head in original coordinates.
    """
    h = heads[token_idx]
    while h != -1 and upos_tags[h] in {"PUNCT", "SYM"}:
        h = heads[h]
    return h

def compute_tree_height(heads: List[int], upos_tags: List[str]) -> Optional[int]:
    """
    Compute the height of the content-only dependency tree.
    Returns max depth from root to any leaf in content tree.
    """
    content_indices = compute_content_indices(upos_tags)

    if len(content_indices) == 0:
        return None

    # Build content-only tree with collapsed heads
    content_tree = {}  # child -> parent mapping in content coordinates
    orig_to_content = {orig_idx: i for i, orig_idx in enumerate(content_indices)}

    for content_idx, orig_idx in enumerate(content_indices):
        content_head_orig = content_head_with_collapse(orig_idx, heads, upos_tags)
        if content_head_orig == -1:
            # Root node
            content_tree[content_idx] = None
        elif content_head_orig in orig_to_content:
            # Content head
            content_tree[content_idx] = orig_to_content[content_head_orig]
        else:
            # Should not happen with proper collapse, but handle gracefully
            content_tree[content_idx] = None

    # Compute depth for each node using DFS
    def compute_depth(node: int, visited: set) -> int:
        if node in visited:
            # Cycle detected, return 0 to avoid infinite recursion
            return 0

        if content_tree[node] is None:
            # Root node
            return 1

        visited.add(node)
        parent_depth = compute_depth(content_tree[node], visited)
        visited.remove(node)

        return parent_depth + 1

    # Find maximum depth
    max_depth = 0
    for node in content_tree:
        depth = compute_depth(node, set())
        max_depth = max(max_depth, depth)

    return max_depth

def compute_sentence_stats(sent_data: Dict[str, Any]) -> SentenceStats:
    """Compute content length and arc length stats for one sentence."""
    tokens = sent_data["tokens"]
    heads = sent_data["head_indices"]  # 0-based, -1 for root
    upos_tags = sent_data["upos_tags"]

    # Get sent_id if available, otherwise use index
    sent_id = sent_data.get("sent_id", f"sent_{sent_data.get('index', 'unknown')}")

    # Basic length metrics
    orig_len_incl_punct = len(tokens)
    content_indices = compute_content_indices(upos_tags)
    content_len = len(content_indices)
    content_ratio = content_len / orig_len_incl_punct if orig_len_incl_punct > 0 else 0.0

    # Evaluability flags
    uuas_evaluable = content_len >= 2
    dist_spear_evaluable = content_len >= 3
    depth_spear_evaluable = content_len >= 2

    # Arc length computation (skip if < 2 content tokens)
    mean_arc_len = None
    num_content_arcs_used = 0

    if content_len >= 2:
        # Map original index -> content-only rank
        rank_map = {orig_i: k for k, orig_i in enumerate(content_indices)}

        arc_lengths = []
        for orig_i in content_indices:
            content_head_idx = content_head_with_collapse(orig_i, heads, upos_tags)

            # Skip if root or head not in content set
            if content_head_idx == -1 or content_head_idx not in rank_map:
                continue

            # Compute content-only distance
            dep_rank = rank_map[orig_i]
            head_rank = rank_map[content_head_idx]
            arc_lengths.append(abs(head_rank - dep_rank))

        num_content_arcs_used = len(arc_lengths)
        if arc_lengths:
            mean_arc_len = sum(arc_lengths) / len(arc_lengths)

    # Tree height computation
    tree_height = compute_tree_height(heads, upos_tags)

    return SentenceStats(
        sent_id=sent_id,
        content_len=content_len,
        mean_arc_len=mean_arc_len,
        tree_height=tree_height,
        num_content_arcs_used=num_content_arcs_used,
        orig_len_incl_punct=orig_len_incl_punct,
        content_ratio=content_ratio,
        uuas_evaluable=uuas_evaluable,
        dist_spear_evaluable=dist_spear_evaluable,
        depth_spear_evaluable=depth_spear_evaluable
    )

def process_language_split(language_slug: str, split: str) -> List[SentenceStats]:
    """Process one language/split and return sentence stats."""
    conllu_path = REPO_ROOT / "data" / "ud" / language_slug / f"{split}.conllu"

    if not conllu_path.exists():
        print(f"Warning: {conllu_path} not found, skipping")
        return []

    print(f"Processing {language_slug}/{split}...")

    try:
        sentences = read_conll_file(str(conllu_path))
        stats = []

        for i, sent_data in enumerate(sentences):
            # Add index for sent_id fallback
            sent_data["index"] = i
            stat = compute_sentence_stats(sent_data)
            stats.append(stat)

        print(f"  Processed {len(stats)} sentences")
        return stats

    except Exception as e:
        print(f"Error processing {conllu_path}: {e}")
        return []

def save_stats(stats: List[SentenceStats], output_path: Path) -> None:
    """Save sentence stats to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for stat in stats:
            json.dump({
                "sent_id": stat.sent_id,
                "content_len": stat.content_len,
                "mean_arc_len": stat.mean_arc_len,
                "tree_height": stat.tree_height,
                "num_content_arcs_used": stat.num_content_arcs_used,
                "orig_len_incl_punct": stat.orig_len_incl_punct,
                "content_ratio": stat.content_ratio,
                "uuas_evaluable": stat.uuas_evaluable,
                "dist_spear_evaluable": stat.dist_spear_evaluable,
                "depth_spear_evaluable": stat.depth_spear_evaluable
            }, f)
            f.write('\n')

def get_available_languages() -> List[str]:
    """Get list of available UD language directories."""
    ud_root = REPO_ROOT / "data" / "ud"
    if not ud_root.exists():
        return []

    return [
        d.name for d in ud_root.iterdir()
        if d.is_dir() and d.name.startswith("UD_")
    ]

def main():
    """Main entry point."""
    languages = get_available_languages()

    if not languages:
        print("No UD languages found in data/ud/")
        return

    print(f"Found {len(languages)} UD languages")

    # Process each language and split
    for language_slug in sorted(languages):
        for split in ["dev", "test"]:
            stats = process_language_split(language_slug, split)

            if stats:
                output_dir = REPO_ROOT / "outputs" / "analysis" / "final" / "01_covariates" / "sentence_stats" / language_slug
                output_path = output_dir / f"{split}_content_stats.jsonl"
                save_stats(stats, output_path)
                print(f"  Saved to {output_path}")

    print("\nSentence covariate computation complete!")

if __name__ == "__main__":
    main()
