"""Frozen-model DEV23 experiment: per-video K equals the Fine interval count.

Only this experiment driver is new. Original selectors, features, seeds, MP4
alignment and scoring are called unchanged. Fine counts are an explicit oracle
budget; Fine boundaries are exported only after every selection is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0',
                  OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                  PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from data import load_inference, read_json, sha256, write_json_new
from run1019 import SEEDS, check_frozen, csvout, load_paths

CORE = ('data.py', 'prepare.py', 'train.py', 'predict.py', 'evaluate.py',
        'vision.py', 'baselines.py', 'run1019.py')
MODEL = ROOT / 'results/DEV23_FINAL_20261020_1019/train/HGKS_LITE_MODEL.joblib'


def verify_freeze(out):
    frozen = read_json(out / 'code_freeze.json')
    for name, expected in frozen['files'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('FROZEN_INPUT_CHANGED ' + name)
    return frozen


def command(args, out, label):
    """Keep exact commands, live progress and exit codes without overwriting."""
    args = list(map(str, args))
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    start = time.monotonic()
    with (logs / (label + '.stdout.txt')).open('x', encoding='utf-8') as stdout, \
         (logs / (label + '.stderr.txt')).open('x', encoding='utf-8') as stderr:
        result = subprocess.run(args, cwd=ROOT, stdout=stdout, stderr=stderr)
    write_json_new(logs / (label + '.command.json'), {
        'args': args, 'cwd': str(ROOT), 'returncode': result.returncode,
        'elapsed_s': time.monotonic() - start})
    if result.returncode:
        raise RuntimeError('COMMAND_FAILED ' + label + '; see ' + str(logs))
    print('PASS', label, flush=True)


def counts(paths, out):
    """A separate process exports integers, never interval boundaries, to selectors."""
    import pandas as pd
    pools, durations, *_ = load_inference(paths['dev'], 'DEV')
    ids = sorted(r['video_id'] for r in read_json(paths['split'])['videos'] if r['split'] == 'DEV')
    if set(ids) != set(pools) or len(ids) != 23:
        raise ValueError('EXPECTED_DEV23')
    source = paths['annotations'] / 'narrations-and-action-segments/HD_EPIC_Narrations.pkl'
    table = pd.read_pickle(source)
    frequencies = table['video_id'].value_counts()
    K = {v: int(frequencies.get(v, 0)) for v in ids}
    if sum(K.values()) != 10344 or any(k < 1 for k in K.values()):
        raise ValueError('EXPECTED_10344_FINE_INTERVALS')
    if any(K[v] > len(pools[v]) for v in ids):
        raise ValueError('LITE_POOL_TOO_SMALL')
    write_json_new(out / 'budgets.json', {
        'K': K, 'K_total': sum(K.values()), 'source_sha256': sha256(source),
        'budget_rule': 'PER_VIDEO_OFFICIAL_FINE_INTERVAL_COUNT',
        'oracle_count_access': True, 'boundary_exported_to_selection': False})
    from prepare import budget
    csvout(out / 'budgets.csv', [dict(video_id=v, duration_s=durations[v],
        old_K=budget(durations[v]), K=K[v], lite_candidates=len(pools[v]),
        visual_candidates=4*K[v]) for v in ids])


def clock_for(paths, v):
    cp = paths['clocks'] / v[:3] / (v + '_mp4_to_vrs_time_ns.csv')
    lp = cp.with_name(v + '_vrs_to_mp4_log.json')
    receipt = read_json(paths['features'] / v / 'FEATURE_RECEIPT.json')
    if sha256(cp) != receipt['video_asset']['CSV_sha256'] or sha256(lp) != receipt['video_asset']['log_sha256']:
        raise ValueError('OFFICIAL_CLOCK_CHANGED')
    clock = np.loadtxt(cp, delimiter=',', skiprows=1, usecols=0, ndmin=1) / 1e9
    if len(clock) != read_json(lp)['num_mp4_frames'] or not np.all(np.diff(clock) > 0):
        raise ValueError('INVALID_OFFICIAL_CLOCK')
    return clock


def mapped_row(times, K, clock):
    from vision import snap_to_mp4
    row = snap_to_mp4(sorted(map(float, times)), clock)
    indices = row['official_frame_indices_zero_based']
    if len(indices) != K or len(set(indices)) != K or any(i < 0 or i >= len(clock) for i in indices):
        raise ValueError('WRONG_K_OR_INVALID_PHYSICAL_FRAMES')
    return {'K': K, **row}


def lite(paths, out):
    import joblib
    import predict
    import sklearn
    if sklearn.__version__ != '1.8.0' or np.__version__ != '2.3.5':
        raise ValueError('LITE_ENVIRONMENT_CHANGED')
    if sha256(MODEL) != read_json(MODEL.parent / 'TRAIN_RECEIPT.json')['model_sha256']:
        raise ValueError('MODEL_RECEIPT_MISMATCH')
    model = joblib.load(MODEL)
    pools, durations, *_ = load_inference(paths['dev'], 'DEV')
    budgets = read_json(out / 'budgets.json')['K']
    selected = {}
    for v, K in sorted(budgets.items()):
        # Replace only the duration->K dependency for this call. The actual
        # frozen select_video function, ranks, coefficients and time_scale run.
        with patch.object(predict, 'budget', return_value=K) as injected:
            rows, diagnostics = predict.select_video(pools[v], model, durations[v])
            injected.assert_called_once_with(durations[v])
        if diagnostics['K'] != K or len(rows) != K:
            raise ValueError('LITE_BUDGET_NOT_APPLIED')
        selected[v] = {'K': K, 'requested_times_s': [r['time_s'] for r in rows],
                       'candidate_indices': [r['candidate_index'] for r in rows]}
        print('LITE', v, K, flush=True)
    write_json_new(out / 'lite_selected.json', {
        'method': '1017_LITE_FINAL', 'videos': selected,
        'model_sha256': sha256(MODEL), 'numpy': np.__version__, 'sklearn': sklearn.__version__,
        'budget_injection': 'SCOPED_PATCH_OF_PREDICT_BUDGET_ONLY',
        'oracle_count_access': True, 'Fine_boundary_access': False})


def dense_features(paths, out, v, times, steps, clock):
    """Reuse the exact original per-request JPEG and offline feature functions."""
    import vision
    folder = out / 'raw_features' / v
    folder.mkdir(parents=True, exist_ok=False)
    source = read_json(paths['video_sources'])[v]
    mp4 = ROOT / source['mp4']
    if sha256(mp4) != source['mp4_sha256']:
        raise ValueError('MP4_SHA_CHANGED')
    with mp4.open('rb') as f:
        if hashlib.file_digest(f, 'md5').hexdigest() != source['official_md5']:
            raise ValueError('MP4_OFFICIAL_MD5_MISMATCH')
    ff = vision.ffmpeg()
    meta = vision.probe(mp4, ff)
    args = [str(paths['ffprobe']), '-v', 'error', '-select_streams', 'v:0',
            '-show_packets', '-show_entries', 'packet=pts_time', '-of', 'csv=p=0', str(mp4)]
    proc = subprocess.run(args, capture_output=True, text=True, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    pts = np.sort([float(x.split(',')[0]) for x in proc.stdout.splitlines() if x.strip()]) - meta['format_start_s']
    if len(pts) != len(clock) or np.max(abs(pts-clock)) > 1.1e-6:
        raise ValueError('MP4_PTS_CLOCK_MISMATCH')

    def decode_one(item):
        i, t = item
        frame_dir = folder / f'{i:05d}'
        row = vision.decode_sequence(ff, mp4, [float(t)], frame_dir, meta['format_start_s'], True)[0]
        row['image_path'] = str(frame_dir / row['jpeg_filename'])
        expected = clock[min(int(np.searchsorted(clock, t-1e-9)), len(clock)-1)]
        if abs(row['actual_relative_s']-expected) > 1.1e-6:
            raise ValueError('JPEG_OFFICIAL_FRAME_MISMATCH')
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=8) as workers:
        for row in workers.map(decode_one, enumerate(times)):
            rows.append(row)
            if len(rows) % 100 == 0 or len(rows) == len(times):
                print('RAW_RGB', v, len(rows), '/', len(times), flush=True)
    write_json_new(folder / 'ORDERED_JPEGS.json', rows)
    h = vision.histograms([r['image_path'] for r in rows])
    print('CLIP_START', v, len(rows), flush=True)
    X, q = vision.encode_clip(rows, steps, paths['clip_snapshot'])
    if h.shape != (len(times), 40) or X.shape != (len(times), 512) or q.shape != (512,):
        raise ValueError('BAD_FEATURE_SHAPE')
    if not all(np.isfinite(a).all() for a in (h, X, q)):
        raise ValueError('NONFINITE_FEATURES')
    cache = folder / 'SHARED_HSV40_CLIP512.npz'
    with cache.open('xb') as f:
        np.savez_compressed(f, histograms=h, frame_features=X, sop_features=q)
    write_json_new(folder / 'FEATURE_RECEIPT.json', {
        'video_id': v, 'mode': 'NEW_RGB_FINE_BUDGET_4K', 'n_candidates': len(times),
        'feature_npz_sha256': sha256(cache), 'ordered_JPEG_sha256': sha256(folder / 'ORDERED_JPEGS.json'),
        'MP4_sha256': source['mp4_sha256'], 'MP4_official_md5': source['official_md5'],
        'PTS_max_error_s': float(np.max(abs(pts-clock))), 'ffprobe_command': args,
        'ffmpeg_sha256': sha256(ff), 'CLIP_model_id': vision.MODEL_ID,
        'SOP_steps': steps, 'Fine_boundary_access': False})
    return h, X, q


def baselines_stage(paths, out):
    import baselines
    import sklearn
    if sklearn.__version__ != '1.9.1' or np.__version__ != '2.4.0':
        raise ValueError('BASELINE_ENVIRONMENT_CHANGED')
    pools, durations, stages, sop, *_ = load_inference(paths['dev'], 'DEV')
    budgets = read_json(out / 'budgets.json')['K']
    lite_requests = read_json(out / 'lite_selected.json')['videos']
    methods = {'1017_LITE_FINAL': {
        v: mapped_row(r['requested_times_s'], budgets[v], clock_for(paths, v))
        for v, r in lite_requests.items()}}
    # Shortest video first provides an early complete pipeline check.
    for v in sorted(budgets, key=lambda v: (budgets[v], v)):
        K = budgets[v]
        clock = clock_for(paths, v)
        times = baselines.grid(durations[v], K)
        recipes = list(dict.fromkeys(s['recipe_id'] for s in stages[v] if s.get('recipe_id')))
        steps = [s for rid in recipes for s in sop[rid]]
        h, X, q = dense_features(paths, out, v, times, steps, clock)
        found = baselines.all_indices(v, times, K, stages[v], h, X, q)
        found['AKS_SOP_CLIP_RECONSTRUCTED_QUERY_ADAPT'] = found.pop('AKS_SOP_CLIP_ADAPT')
        requests = {m: times[ix].tolist() for m, ix in found.items() if m != 'RANDOM_K'}
        requests['UNIFORM_FULL_ALL_FRAMES'] = clock[baselines.uniform_indices(len(clock), K)].tolist()
        pool_times = np.array([c['t'] for c in pools[v]])
        requests['UNIFORM_POOL_CURRENT'] = pool_times[baselines.uniform_pool_indices(pool_times, K)].tolist()
        for seed in SEEDS:
            for domain, arr in [('4K', times), ('ALL_FRAMES', clock)]:
                requests[f'RANDOM_{domain}_SEED_{seed}'] = arr[baselines.random_indices(
                    len(arr), K, seed + sum(map(ord, v)))].tolist()
        local = {m: mapped_row(ts, K, clock) for m, ts in requests.items()}
        write_json_new(out / 'per_video_selections' / (v + '.json'), local)
        for m, row in local.items():
            methods.setdefault(m, {})[v] = row
        print('ALL_BASELINES', v, K, len(local), flush=True)
    if len(methods) != 30 or any(set(rows) != set(budgets) for rows in methods.values()):
        raise ValueError('EXPECTED_30_METHOD_SEEDS_X_DEV23')
    manifest = {'oracle_count_access': True, 'Fine_boundary_access': False, 'methods': {}}
    for m, rows in sorted(methods.items()):
        selection = out / 'frozen_selections' / (m + '.json')
        transport = out / 'native' / m / 'DEV_METHOD_FROZEN.json'
        write_json_new(selection, {'method_id': m, 'videos': rows, 'oracle_count_access': True,
            'Fine_boundary_access': False, 'budget_rule': 'PER_VIDEO_OFFICIAL_FINE_INTERVAL_COUNT'})
        write_json_new(transport, {'schema': 'HGKS_FINAL_PREDICTION_V1', 'split': 'DEV',
            'predictions': {v: {'K': r['K'], 'selected': [{'time_s': t} for t in r['selected_times_s']]}
                            for v, r in rows.items()}, 'source_selection_sha256': sha256(selection)})
        receipt = transport.parent / 'DEV_SELECTION_RECEIPT.json'
        write_json_new(receipt, {'prediction_sha256': sha256(transport), 'oracle_count_access': True,
                                 'Fine_boundary_access': False})
        manifest['methods'][m] = dict(selection=str(selection.relative_to(out)),
            selection_sha256=sha256(selection), transport=str(transport.relative_to(out)),
            transport_sha256=sha256(transport), receipt_sha256=sha256(receipt),
            n_videos=len(rows), K_total=sum(r['K'] for r in rows.values()))
    write_json_new(out / 'baseline_environment.json', {
        'numpy': np.__version__, 'sklearn': sklearn.__version__, 'executable': sys.executable})
    write_json_new(out / 'frozen_manifest.json', manifest)


def evaluate_stage(paths, out, tolerance_s=0.0):
    manifest = check_frozen(out)
    first = out / manifest['methods']['1017_LITE_FINAL']['transport']
    gold = out / 'gold/DEV23_FINE.json'
    command([sys.executable, '-B', '-X', 'utf8', ROOT / 'evaluate.py', 'targets',
             '--predictions', first, '--annotations', paths['annotations'], '--out', gold], out, 'targets')
    counts_doc = read_json(out / 'budgets.json')
    target_doc = read_json(gold)
    if target_doc['source_sha256'] != counts_doc['source_sha256'] or \
       {v: len(rows) for v, rows in target_doc['targets'].items()} != counts_doc['K']:
        raise ValueError('COUNT_AND_EVALUATION_TARGETS_DIFFER')
    stats = {}
    for m, row in manifest['methods'].items():
        pred = out / row['transport']
        command([sys.executable, '-B', '-X', 'utf8', ROOT / 'evaluate.py', 'score',
                 '--predictions', pred, '--targets', gold, '--out', pred.parent / 'evaluation.json',
                 '--tolerance', tolerance_s], out, 'score_' + m)
        stats[m] = read_json(pred.parent / 'evaluation.json')['time_interval_metrics']
    lite_stats = stats['1017_LITE_FINAL']
    rows, per_video = [], []
    for m, r in stats.items():
        if r['total_intervals'] != 10344 or manifest['methods'][m]['K_total'] != 10344:
            raise ValueError('WRONG_TOTAL_FINE_OR_K')
        rows.append(dict(method_id=m, hits=r['total_hits'], Fine_intervals=r['total_intervals'],
            K_total=10344, Micro_percent=100*r['micro'], Macro_percent=100*r['macro'],
            delta_Micro_pp_vs_Lite=100*(r['micro']-lite_stats['micro'])))
        for v, vr in r['by_video'].items():
            per_video.append(dict(method_id=m, video_id=v, K=counts_doc['K'][v],
                Fine_intervals=vr['n_intervals'], hits=vr['hits'], rate_percent=100*vr['rate'],
                delta_hits_vs_Lite=vr['hits']-lite_stats['by_video'][v]['hits']))
    csvout(out / 'all_seed_results.csv', rows)
    csvout(out / 'per_video.csv', per_video)
    summary = [{**r, 'Micro_std_percent': 0., 'Macro_std_percent': 0., 'n_seeds': 1}
               for r in rows if not r['method_id'].startswith('RANDOM_')]
    for family in ['RANDOM_4K', 'RANDOM_ALL_FRAMES']:
        seeds = [r for r in rows if r['method_id'].startswith(family + '_SEED_')]
        if len(seeds) != 10:
            raise ValueError('EXPECTED_TEN_SEEDS')
        row = dict(seeds[0])
        row.update({key: float(np.mean([r[key] for r in seeds])) for key in
                    ('hits', 'Micro_percent', 'Macro_percent', 'delta_Micro_pp_vs_Lite')})
        row.update(method_id=family+'_10_SEEDS', n_seeds=10,
            Micro_std_percent=float(np.std([r['Micro_percent'] for r in seeds], ddof=0)),
            Macro_std_percent=float(np.std([r['Macro_percent'] for r in seeds], ddof=0)))
        summary.append(row)
    csvout(out / 'summary.csv', summary)
    verify_freeze(out)
    check_frozen(out)
    write_json_new(out / 'RESULT.json', {'status': 'PASS', 'n_videos': 23, 'n_method_seeds': 30,
        'K_total_per_method': 10344, 'Fine_intervals': 10344, 'visual_candidates': 41376,
        'metric': f'Fine Action one-hit +/-{tolerance_s:g}s',
        'tolerance_seconds': tolerance_s, 'core_source_and_model_unchanged': True,
        'comparison_scope': 'SYSTEM_LEVEL_SAME_DEV_SAME_ORACLE_COUNT_BUDGET',
        'training_budget_unchanged': True, 'retrained': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['run', 'counts', 'lite', 'baselines', 'evaluate'])
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--tolerance', type=float, choices=(0.0, 0.5), default=0.0,
                   help='Evaluation tolerance in seconds: default 0, optional 0.5')
    a = p.parse_args()
    out = a.out.resolve()
    paths = load_paths(ROOT / 'paths.json')
    if a.stage == 'run':
        out.mkdir(parents=True, exist_ok=False)
        names = [*CORE, 'tests/compare_fine_budget.py', 'paths.json',
                 str(MODEL.relative_to(ROOT)), str((MODEL.parent/'TRAIN_RECEIPT.json').relative_to(ROOT)),
                 str(paths['dev'].relative_to(ROOT)), str(paths['split'].relative_to(ROOT)),
                 'models/clip-vit-base-patch32/pytorch_model.bin']
        # Some local snapshots use safetensors rather than pytorch_model.bin.
        names = [n for n in names if (ROOT/n).is_file()]
        names += [str(f.relative_to(ROOT)) for f in paths['clip_snapshot'].glob('*.safetensors')]
        write_json_new(out / 'code_freeze.json', {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'files': {n: sha256(ROOT/n) for n in names}, 'retrain': False,
            'oracle_count_access': True, 'Fine_boundaries_used_for_selection': False,
            'invocation': sys.argv, 'executable': sys.executable})
        for stage, interpreter in [('counts', sys.executable), ('lite', sys.executable),
                                   ('baselines', paths['feature_python']), ('evaluate', sys.executable)]:
            stage_args = [interpreter, '-B', '-X', 'utf8', Path(__file__), stage, '--out', out]
            if stage == 'evaluate':
                stage_args += ['--tolerance', a.tolerance]
            command(stage_args, out, stage)
        print('COMPLETE', out, flush=True)
        return
    verify_freeze(out)
    if a.stage == 'evaluate':
        evaluate_stage(paths, out, a.tolerance)
    else:
        {'counts': counts, 'lite': lite, 'baselines': baselines_stage}[a.stage](paths, out)
    verify_freeze(out)


if __name__ == '__main__':
    main()
