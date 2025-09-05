#!/usr/bin/env python3
"""
Stage 9 post-processing:
1) Flag low-coverage rows for the L<=20 anchor and summarize per-language.
2) Build master_additions_stage9_test.csv with anchor columns for distance/depth.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> None:
    root = Path('outputs/analysis')
    find = root / 'stage9_findings'
    find.mkdir(parents=True, exist_ok=True)

    # 1) Low-coverage under L<=20
    len_df = pd.read_csv(root / 'anchored_eval_len_per_layer_test.csv')
    low = len_df[len_df['low_coverage_flag']].copy()
    keep_cols = [
        'language_slug', 'probe', 'layer',
        'coverage_all', 'coverage_base',
        'T_eligible', 'T_base', 'N_total',
    ]
    low_out = low[keep_cols].sort_values(['language_slug', 'probe', 'layer'])
    low_out.to_csv(find / 'low_coverage_L_anchor_rows.csv', index=False)

    lang_summary = (
        low.groupby('language_slug')
        .agg(
            n_flag_rows=('language_slug', 'size'),
            min_coverage_all=('coverage_all', 'min'),
            median_coverage_all=('coverage_all', 'median'),
        )
        .reset_index()
        .sort_values('n_flag_rows', ascending=False)
    )
    lang_summary.to_csv(find / 'low_coverage_L_anchor_by_language.csv', index=False)

    # 2) Master additions for Stage 9 (test)
    arc_df = pd.read_csv(root / 'anchored_eval_arclen_per_layer_test.csv')

    def extract_anchor(df: pd.DataFrame, is_len: bool) -> pd.DataFrame:
        rows = []
        for probe, metric_prefix in [('dist', 'uuas'), ('depth', 'rootacc')]:
            sub = df[df['probe'] == probe][
                ['language_slug', 'probe', 'layer', 'point_estimate', 'ci_low', 'ci_high', 'coverage_all']
            ].copy()
            if is_len:
                sub = sub.rename(
                    columns={
                        'point_estimate': f'{metric_prefix}_at_L20',
                        'ci_low': f'{metric_prefix}_at_L20_ci_low',
                        'ci_high': f'{metric_prefix}_at_L20_ci_high',
                        'coverage_all': f'{metric_prefix}_L20_coverage_all',
                    }
                )
            else:
                sub = sub.rename(
                    columns={
                        'point_estimate': f'{metric_prefix}_at_A3',
                        'ci_low': f'{metric_prefix}_at_A3_ci_low',
                        'ci_high': f'{metric_prefix}_at_A3_ci_high',
                        'coverage_all': f'{metric_prefix}_A3_coverage_all',
                    }
                )
            rows.append(sub)
        out = rows[0].merge(rows[1], on=['language_slug', 'probe', 'layer'], how='outer')
        return out

    len_add = extract_anchor(len_df, is_len=True)
    arc_add = extract_anchor(arc_df, is_len=False)
    add = len_add.merge(arc_add, on=['language_slug', 'probe', 'layer'], how='outer')

    out_add = root / 'master_additions_stage9_test.csv'
    add.to_csv(out_add, index=False)
    print('✓ Wrote', out_add)
    print('✓ Low-coverage files:',
          find / 'low_coverage_L_anchor_rows.csv',
          find / 'low_coverage_L_anchor_by_language.csv')


if __name__ == '__main__':
    main()


