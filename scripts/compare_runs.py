"""
Paired bootstrap comparison of two evaluations on the same test images.

    python scripts/compare_runs.py results/per_image_uni_phase2_lora.json \
        results/baseline_imagenet_fallback/per_image_imagenet_fallback_phase2_lora_hovernet_pp.json \
        --out results/comparison_uni_vs_imagenet_phase2.json

Inputs are the per_image_<tag>.json files written by evaluate.py; compare files made
with the same post-processing (meta.postproc; files without it used 'notebook'). Test images are
resampled with replacement (paired: the same resample is applied to both runs) and
the pooled scores are recomputed on every resample exactly as InstanceSegEvaluator
pools them over the whole set (F1 = RQ = 2TP / (2TP + FP + FN), PQ = sum of matched
IoU / (TP + FP/2 + FN/2), foreground IoU from summed pixel counts). Reported:
A, B, A - B, the 95 % percentile interval of A - B and the share of resamples
in which A is not better than B (A - B <= 0, or >= 0 for count MAE), a one-sided
bootstrap p-value for "A is better".

The interval reflects test-image sampling only. It says nothing about training
variance (seed, data order), which a single run per configuration cannot measure.
"""
import argparse
import json

import numpy as np

# name -> (higher_is_better, function of summed counts)
METRICS = {
    'Foreground IoU': (True, lambda s: s['fg_tp'] / (s['fg_tp'] + s['fg_fp'] + s['fg_fn'])),
    'Precision @0.5': (True, lambda s: s['tp'] / (s['tp'] + s['fp'])),
    'Recall @0.5': (True, lambda s: s['tp'] / (s['tp'] + s['fn'])),
    'F1 @0.5 (= RQ)': (True, lambda s: 2 * s['tp'] / (2 * s['tp'] + s['fp'] + s['fn'])),
    'SQ': (True, lambda s: s['sum_iou'] / s['tp']),
    'PQ (standard)': (True, lambda s: s['sum_iou'] / (s['tp'] + 0.5 * s['fp'] + 0.5 * s['fn'])),
    'Count MAE (cells/image)': (False, lambda s: s['abs_count_err'] / s['n_images']),
}
KEYS = ['n_gt', 'n_pred', 'tp', 'fp', 'fn', 'sum_iou', 'fg_tp', 'fg_fp', 'fg_fn']


def load(path):
    with open(path) as f:
        d = json.load(f)
    arr = {k: np.array([s[k] for s in d['stats']], dtype=np.float64) for k in KEYS}
    arr['abs_count_err'] = np.abs(arr['n_pred'] - arr['n_gt'])
    arr['n_images'] = np.ones(len(d['stats']))
    return d, arr


def pooled(arr, idx):
    """Sum per-image counts over resample(s). idx: (n,) or (B, n) image indices."""
    return {k: v[idx].sum(axis=-1) for k, v in arr.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('a', help='per_image_<tag>.json of run A')
    p.add_argument('b', help='per_image_<tag>.json of run B (the reference)')
    p.add_argument('--n-boot', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None, help='write the comparison as JSON here')
    args = p.parse_args()

    da, a = load(args.a)
    db, b = load(args.b)
    if da['images'] != db['images']:
        raise SystemExit('The two files do not cover the same images in the same order.')
    n = len(da['images'])

    rng = np.random.default_rng(args.seed)
    full = np.arange(n)
    rows = {}
    for chunk_start in range(0, args.n_boot, 1000):
        idx = rng.integers(0, n, size=(min(1000, args.n_boot - chunk_start), n))
        sa, sb = pooled(a, idx), pooled(b, idx)
        for name, (_, fn) in METRICS.items():
            rows.setdefault(name, []).append(fn(sa) - fn(sb))

    result = {'a': da['tag'], 'b': db['tag'], 'n_images': n, 'n_boot': args.n_boot, 'seed': args.seed,
              'metrics': {}}
    print(f"A = {da['tag']}   B = {db['tag']}   ({n} paired test images, {args.n_boot} bootstrap resamples)\n")
    print(f"| Metric | A | B | A - B | 95% CI of A - B | P(A not better) |")
    print(f"|---|--:|--:|--:|--:|--:|")
    for name, (higher_better, fn) in METRICS.items():
        va, vb = fn(pooled(a, full)), fn(pooled(b, full))
        diffs = np.concatenate(rows[name])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_not_better = float((diffs <= 0).mean() if higher_better else (diffs >= 0).mean())
        result['metrics'][name] = {'a': float(va), 'b': float(vb), 'diff': float(va - vb),
                                   'ci95': [float(lo), float(hi)], 'p_a_not_better': p_not_better,
                                   'higher_is_better': higher_better}
        print(f'| {name} | {va:.4f} | {vb:.4f} | {va - vb:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p_not_better:.4f} |')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
