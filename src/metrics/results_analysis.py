#!/usr/bin/env python3
"""Results analysis for the ICASSP paper, organized by Peggy's six experiment branches.

WHAT YOU NEED (all downloads, no re-runs):
  1. The run_3 worksheet exported as CSV:
       Google Sheet -> select the run_3 tab -> File -> Download -> CSV
     Save it as run_3.csv
  2. (Optional, for the adult/child table) a per-phone CSV with columns
       spk, fn, phone, loss, score  (and z if calibration was applied)
     — this is exactly what your metrics script builds internally before computing
     metrics; add one df.to_csv() line there to dump it. No model re-run involved.
  3. (Optional, same table) scores_enhanced.json for the age field
     — rebuild it after fixing the metadata.py unpacking bug, then spot-check ages.

USAGE:
  python3 results_analysis.py --run3 run_3.csv
  python3 results_analysis.py --run3 run_3.csv --perphone twist_phone.csv --ages scores_enhanced.json

Every table is labeled with the branch it supports and the sentence it feeds.
"""
import argparse, json, sys
import numpy as np
import pandas as pd

pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda v: f'{v:0.3f}')


def load_run3(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # normalize column names across possible variants
    ren = {'auc_1': 'auc1', 'auc_2': 'auc2', 'sample_size': 'n'}
    df = df.rename(columns=ren)
    for col in ['pcc', 'auc1', 'auc2', 'n']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['calibration', 'normalization']:
        if col in df:
            df[col] = df[col].astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])
    for col in ['model_type', 'model_name', 'granularity', 'pooling']:
        if col in df:
            df[col] = df[col].astype(str).str.strip()
    return df


def show(title, frame, note=''):
    print('\n' + '=' * 78)
    print(title)
    if note:
        print(note)
    print('-' * 78)
    print(frame.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run3', required=True)
    ap.add_argument('--perphone')
    ap.add_argument('--ages')
    args = ap.parse_args()

    df = load_run3(args.run3)
    base = df[(~df.calibration) & (~df.normalization)]

    # ---- Branch 1: localization collapse (raw pipeline, mean pooling) ----
    b1 = base[base.pooling.isin(['mean', 'none'])].pivot_table(
        index='model_name', columns='granularity', values='auc1', aggfunc='first')
    b1 = b1[[c for c in ['phone', 'word', 'utterance'] if c in b1.columns]]
    show('BRANCH 1 — Localization collapse: AUC by model x granularity '
         '(uncalibrated, unnormalized, mean pooling)', b1,
         'Feeds: "utterance-level signal exists but does not localize."')

    b1p = base[base.pooling.isin(['mean', 'none'])].pivot_table(
        index='model_name', columns='granularity', values='pcc', aggfunc='first')
    show('BRANCH 1 (companion) — PCC, same cells', b1p,
         'Convention: informative = negative (loss vs human score).')

    # ---- Branch 2a: pooling ablation (phone level, uncalibrated) ----
    b2 = base[base.granularity == 'phone'].pivot_table(
        index='model_name', columns='pooling', values='auc1', aggfunc='first')
    show('BRANCH 2 — Pooling ablation: phone-level AUC by pooling method', b2,
         'Feeds: one ablation paragraph; expect mean ~ >= max, std unstable.')

    # ---- Branch 2b: frequency normalization on/off ----
    rows = []
    for (m, g, p), grp in df[~df.calibration].groupby(['model_name', 'granularity', 'pooling']):
        off = grp[~grp.normalization].auc1.dropna()
        on = grp[grp.normalization].auc1.dropna()
        if len(off) and len(on):
            rows.append({'model': m, 'gran': g, 'pool': p,
                         'auc_raw': off.iloc[0], 'auc_norm': on.iloc[0],
                         'delta': on.iloc[0] - off.iloc[0]})
    b2b = pd.DataFrame(rows)
    if len(b2b):
        show('BRANCH 3 — Frequency normalization effect (AUC with minus without)',
             b2b.sort_values(['model', 'gran', 'pool']).set_index(['model', 'gran', 'pool']),
             'Feeds: one robustness sentence; near-zero deltas = "insensitive to freq norm."')

    # ---- Branch 4: enrollment calibration on/off ----
    rows = []
    for (m, g, p), grp in df[~df.normalization].groupby(['model_name', 'granularity', 'pooling']):
        off = grp[~grp.calibration].auc1.dropna()
        on = grp[grp.calibration].auc1.dropna()
        if len(off) and len(on):
            margin = off.iloc[0] - 0.5
            kept = 100 * (on.iloc[0] - 0.5) / margin if margin > 0.01 else np.nan
            rows.append({'model': m, 'gran': g, 'pool': p,
                         'auc_pooled': off.iloc[0], 'auc_calibrated': on.iloc[0],
                         'margin_kept_pct': kept})
    b4 = pd.DataFrame(rows)
    if len(b4):
        b4 = b4[b4.gran.isin(['phone', 'word'])]
        show('BRANCH 5 — Enrollment calibration: pooled vs calibrated AUC '
             '(margin_kept = % of the above-chance margin surviving calibration)',
             b4.sort_values(['model', 'gran', 'pool']).set_index(['model', 'gran', 'pool']),
             'Feeds the headline: "most of the apparent localized signal was speaker confound."')

    # ---- Branch 3: TTS / CosyVoice embedding conditions ----
    cos = df[df.model_type.str.upper().str.contains('COSY')]
    if len(cos):
        b3 = cos[(~cos.normalization)].pivot_table(
            index=['model_name', 'calibration'], columns='granularity',
            values='auc1', aggfunc='first')
        show('BRANCH 4 — Text-conditioned (CosyVoice) by embedding condition', b3,
             'Feeds: "text conditioning through TTS likelihood does not rescue localization; '
             'embedding manipulation barely moves it."')

    # ---- Subgroup: adult vs child (needs per-phone CSV + ages) ----
    if args.perphone and args.ages:
        pp = pd.read_csv(args.perphone, dtype={'spk': str, 'fn': str})
        enh = json.load(open(args.ages))
        pp['age'] = pp.fn.map({k: enh[k].get('age') for k in enh})
        pp = pp.dropna(subset=['age'])
        pp['child'] = pp.age < 18
        pp['bad'] = (pp.score <= 1).astype(int)
        from sklearn.metrics import roc_auc_score
        rows = []
        for name, g in pp.groupby('child'):
            row = {'group': 'child' if name else 'adult', 'n': len(g),
                   'pct_bad': 100 * g.bad.mean(),
                   'auc_pooled': roc_auc_score(g.bad, g.loss)}
            if 'z' in g:
                gz = g.dropna(subset=['z'])
                row['auc_calibrated'] = roc_auc_score(gz.bad, gz.z)
            rows.append(row)
        show('SUBGROUP — Adult vs child (from per-phone data, not the sheet)',
             pd.DataFrame(rows).set_index('group'),
             'Feeds: "the residual signal lives in the children." '
             'Verify ages against official speaker info after fixing metadata.py.')

    # ---- Anchor numbers ----
    print('\n' + '=' * 78)
    print('ANCHOR NUMBERS to recite (fill from the tables above):')
    print(' 1. Phone AUC pooled -> calibrated (best model): the confound headline')
    print(' 2. Adult vs child calibrated AUC: where the residual lives')
    print(' 3. Localization top-1 vs random (19% vs 13.3% from the separate analysis)')
    print(' 4. CosyVoice conditions all ~0.52: conditioning does not rescue')
    print(' 5. Rehearsal 60/60 paired-ratio vs 43% transcription survival')
    print('\nEXCLUSION LIST (state visibly in the paper):')
    print(' - Flow-SLM per-frame values before draw-averaging (single random draw)')
    print(' - TASLM utterance rows pending the segmentation A/B diagnosis')


if __name__ == '__main__':
    main()





