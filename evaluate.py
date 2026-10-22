"""仅在预测冻结后评价：先从官方 Narrations 导出 Gold，再计算命中率。
python evaluate.py targets --help
python evaluate.py score --help
本文件不被 train.py 或 predict.py 导入。
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
from data import read_json, sha256, write_json_new, load_inference
from prepare import budget


# 1. 评价公式：每个动作区间最多记一次命中
def hit_indicators(timestamps, intervals, tolerance_s=0.0):
    if tolerance_s < 0:
        raise ValueError('TOLERANCE_MUST_BE_NONNEGATIVE')
    ts = [float(t) for t in timestamps]
    if any((not float(t) == float(t) for t in ts)):
        raise ValueError('NONFINITE_TIMESTAMP')
    hits = []
    for x in intervals:
        a, b = (float(x['a']), float(x['b']))
        if b < a:
            raise ValueError('INVALID_ACTION_INTERVAL')
        hits.append(int(any((a - tolerance_s <= t <= b + tolerance_s for t in ts))))
    return hits

def one_hit_rate(timestamps, intervals, tolerance_s=0.0):
    hits = hit_indicators(timestamps, intervals, tolerance_s)
    if not hits:
        return {'n_intervals': 0, 'hits': 0, 'rate': None}
    return {'n_intervals': len(hits), 'hits': sum(hits), 'rate': sum(hits) / len(hits)}

def multi_video_report(selected_by_video, target_intervals_by_video, tolerance_s=0.0):
    if set(selected_by_video) != set(target_intervals_by_video):
        raise ValueError('SELECTION_AND_TARGET_VIDEO_SETS_DIFFER')
    each = {}
    all_hits, all_intervals = (0, 0)
    rates = []
    for video_id in sorted(selected_by_video):
        stats = one_hit_rate(selected_by_video[video_id], target_intervals_by_video[video_id], tolerance_s)
        each[video_id] = stats
        all_hits += stats['hits']
        all_intervals += stats['n_intervals']
        if stats['rate'] is not None:
            rates.append(stats['rate'])
    return {'by_video': each, 'micro': all_hits / all_intervals if all_intervals else None,
        'macro': sum(rates) / len(rates) if rates else None, 'total_hits': all_hits,
        'total_intervals': all_intervals, 'tolerance_seconds': tolerance_s,
        'metric_scope': 'action_QA_time_interval_one_hit_not_RGB_or_BT_accuracy'}

def frozen_times(doc):
    if doc.get('schema') != 'HGKS_FINAL_PREDICTION_V1':
        raise ValueError('PREDICTION_FORMAT_UNEXPECTED')
    return {v: [r['time_s'] for r in row['selected']] for v, row in doc['predictions'].items()}

def compare_times(current, reference):
    if set(current) != set(reference):
        raise ValueError('COMPARISON_VIDEO_SETS_DIFFER')
    rows = {}
    for vid in sorted(current):
        a = sorted(map(float, current[vid]))
        b = sorted(map(float, reference[vid]))
        rows[vid] = {'K_current': len(a), 'K_reference': len(b), 'same_K': len(a) == len(b),
            'exact_match': a == b, 'n_shared_exact': len(set(a) & set(b)),
            'max_ordered_difference_s': max((abs(x - y) for x, y in zip(a, b)),
            default=0.0) if len(a) == len(b) else None}
    return {'by_video': rows, 'exact_match_videos': sum((r['exact_match'] for r in rows.values())),
        'all_same_K': all((r['same_K'] for r in rows.values())),
        'all_timestamps_identical': all((r['exact_match'] for r in rows.values()))}

def score_command():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--input', type=Path, help='可选合法推断输入，检查池内成员及K')
    p.add_argument('--targets', type=Path, help='仅预测冻结后才加载的独立Gold')
    p.add_argument('--reference', type=Path, help='原HGKS_FINAL_PREDICTION_V1或明确的{video_id:[time_s]}')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tolerance', type=float, choices=(0.0, 0.5), default=0.0,
                   help='区间外扩容差（秒）：默认0，可选0.5')
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit('REFUSING_TO_OVERWRITE ' + str(args.out))
    d = read_json(args.predictions)
    selected = frozen_times(d)
    receiptpath = args.predictions.parent / (d['split'] + '_SELECTION_RECEIPT.json')
    if not receiptpath.is_file():
        raise ValueError('FROZEN_SELECTION_RECEIPT_REQUIRED')
    if read_json(receiptpath)['prediction_sha256'] != sha256(args.predictions):
        raise ValueError('FROZEN_PREDICTION_CHANGED')
    for vid, ts in selected.items():
        if not all((math.isfinite(t) for t in ts)) or len(ts) != len(set(ts)):
            raise ValueError('INVALID_FROZEN_TIMESTAMPS ' + vid)
        if len(ts) != d['predictions'][vid]['K']:
            raise ValueError('FROZEN_K_INCONSISTENT')
    result = {'prediction_sha256': sha256(args.predictions), 'n_videos': len(selected),
        'n_timestamps': sum(map(len, selected.values())), 'evaluation_status': 'GOLD_NOT_AVAILABLE'}
    if args.input:
        pools, durations, _, _, _, _ = load_inference(args.input, d['split'])
        if set(pools) != set(selected):
            raise ValueError('INPUT_PREDICTION_IDS_DIFFER')
        if sha256(args.input) != d['input_sha256']:
            raise ValueError('PREDICTION_INPUT_CHANGED')
        for vid, ts in selected.items():
            if len(ts) != budget(durations[vid]) or not set(ts) <= {c['t'] for c in pools[vid]}:
                raise ValueError('PREDICTION_NOT_FIXED_K_FROM_POOL ' + vid)
        result['input_quality'] = 'PASS_FIXED_K_UNIQUE_POOL_TIMESTAMPS'
    if args.targets:
        g = read_json(args.targets)
        target = g.get('targets', g)
        result['time_interval_metrics'] = multi_video_report(selected, target, args.tolerance)
        result.update(evaluation_status='POST_FREEZE_TIME_INTERVAL_EVALUATED',
            target_sha256=sha256(args.targets))
    if args.reference:
        ref = read_json(args.reference)
        reference = frozen_times(ref) if ref.get('schema') == 'HGKS_FINAL_PREDICTION_V1' else ref
        if not isinstance(reference, dict) or not all((isinstance(v, list) and all((isinstance(t,
            (int, float)) for t in v)) for v in reference.values())):
            raise ValueError('REFERENCE_SCHEMA_NOT_SUPPORTED; provide explicit {video_id:[time_s]}')
        result['timestamp_comparison'] = compare_times(selected, reference)
        result['reference_sha256'] = sha256(args.reference)
    write_json_new(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

# 2. 冻结预测SHA以后，提取Gold
def targets_command() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True, type=Path)
    p.add_argument('--annotations', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    prediction = read_json(a.predictions)
    split = prediction['split']
    receipt = a.predictions.parent / (split + '_SELECTION_RECEIPT.json')
    if read_json(receipt)['prediction_sha256'] != sha256(a.predictions):
        raise ValueError('Predictions changed before post-hoc evaluation')
    videos = set(prediction['predictions'])
    file = a.annotations / 'narrations-and-action-segments/HD_EPIC_Narrations.pkl'
    table = pd.read_pickle(file)
    targets = {vid: [] for vid in videos}
    for row in table.itertuples(index=False):
        if row.video_id in videos:
            targets[row.video_id].append({'a': float(row.start_timestamp), 'b': float(row.end_timestamp)})
    write_json_new(a.out, {'targets': targets, 'source_sha256': sha256(file),
        'prediction_sha256_at_target_export': sha256(a.predictions), 'scope': split,
        'source': 'OFFICIAL_HD_EPIC_NARRATIONS_POST_PREDICTION_ONLY', 'source_file': str(file.resolve()),
        'prediction_file': str(a.predictions.resolve()), 'selection_receipt_file': str(receipt.resolve())})
    print(f'[POST-PREDICTION TARGETS] {len(targets)} videos -> {a.out}')

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('score','targets'):
        raise SystemExit('用法: python evaluate.py score|targets --help')
    mode=sys.argv[1]
    sys.argv=[sys.argv[0], *sys.argv[2:]]
    if mode == 'score':
        score_command()
    else:
        targets_command()

if __name__ == '__main__':
    main()
