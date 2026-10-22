"""准备步骤（按本文件自上而下阅读）：
1. 手/物体观测每 0.5 秒聚合，明确构造 F0–F15。
2. 16维 → 64维 → 141维，为单个HGB构造训练与推断特征。
3. raw: 官方 RLE、时钟、Object → 各视频候选。
4. inputs: 打包TRAIN Fine监督；DEV/TEST沿用原始输入格式（携带但不使用SOP文本）。

命令：python prepare.py raw --help
      python prepare.py inputs --help
"""
from __future__ import annotations
import argparse, csv, gzip, json, math, zipfile, re, collections, unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from data import (Stop, SCHEMA, read_json, read_jsonl, sha256, write_json_new,
                  write_jsonl_new, decode_hand_video, normalize_official_objects)

# 模型文件中的旧特征身份保留，确保已有本地模型仍可读。
FEATURE_VERSION='HGKS_FRAME_141_FINAL_V1'


# A. 常数和固定帧预算


def budget(duration_s):
    import math
    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('INVALID_VIDEO_DURATION')
    return max(1, math.ceil(duration / 30.0))


# B. 0.5秒聚合 → F0–F15
DT = 0.5

FEATURES = ('hand_speed', 'hand_speed_delta', 'hand_visibility', 'hand_turn', 'left_activity',
    'right_activity', 'object_onset', 'object_offset', 'object_active', 'object_duration_log',
    'joint_temporal', 'hand_rise', 'hand_fall', 'local_steadiness', 'cue_hand', 'cue_object')


def val(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0

def make_candidates(video_id, hand_file, obj_events, duration_s, trace=None):  #这里尝试把手部和物体事件聚合成候选
    T = float(duration_s)    
    n = int(math.ceil(T / DT)) + 2   #时间桶的总数，目前是计算0.5秒内的时间桶总数
    speed = np.zeros((n, 2), dtype=np.float32)   #这里的n是时间桶的数量，这里是个二维数组，这里一般是6行2列，例如speed[2, 0] = 0.8  # 第2桶，左手 speed[2, 1] = 0.3  # 第2桶，右手  可以写成 [0.8, 0.3]
    delta = np.zeros((n, 2), dtype=np.float32)
    direction = np.zeros((n, 2), dtype=np.float32)
    counts = np.zeros((n, 2), dtype=np.int32)
    visible = np.zeros((n, 2), dtype=np.float32)  #记录左右手的可见程度
    with gzip.open(hand_file, 'rt', encoding='utf-8-sig', newline='') as f:  #gzip.open()：读取 .gz 压缩文件。'rt'：解压后按文本读取。
        for r in csv.DictReader(f):
            if r.get('video_id') != video_id:
                raise RuntimeError('CROSS_VIDEO_HAND_DATA ' + video_id)
            t = val(r.get('time_s'))  #这里是从CSV某一行读取 time_s 字段。 val是把数值转化为浮点数
            if not 0 <= t < T:
                continue
            side = {'left': 0, 'right': 1}.get(r.get('hand'))  #先确定左右手
            if side is None:
                raise RuntimeError('UNKNOWN_HAND_SIDE')
            ix = min(n - 1, int(t / DT))  #这里在计算桶的下标。当前的最后时间必须在这个时间除以间隔的桶范围内，不能超过，确保和下角标对齐
            counts[ix, side] += 1
            if str(r.get('visible', '')).lower() == 'true':
                visible[ix, side] += 1
            speed[ix, side] = max(speed[ix, side], min(10.0, val(r.get('speed_norm_per_s'))))   #读取这个时间桶内的最大值，但是不能超过10
            delta[ix, side] = max(delta[ix, side], min(20.0, abs(val(r.get('speed_change_norm_per_s')))))  #读取时间桶内最大的速率差绝对值，含义为帧和帧之间发生过多大的速率变化，最高20
            direction[ix, side] = val(r.get('direction_rad'))  #读取时间桶内最后一次的方向
    coverage = int(np.count_nonzero(counts.sum(axis=1)))
    if coverage < min(n // 4, 10):
        raise RuntimeError('NO_VALID_VERIFIED_HAND_STREAM ' + video_id)
    hs = np.clip(speed.max(axis=1), 0, 4)    #这里是指每一行最大的数值，按照数组的定义就是左右手最大的速度
    ch = np.clip(delta.max(axis=1), 0, 8)    #同理
    dv = np.abs(np.diff(hs, prepend=hs[0]))   #np.diff()：后一个数减前一个数，越大代表两个相邻的时间桶的手变化越大
    vv = (visible.sum(axis=1) / np.maximum(1, counts.sum(axis=1))).clip(0, 1)  #np.maximum(1, counts.sum(axis=1）这里计算手部出现的次数总数除以总桶数来获得手部出现的频率。
    turn = np.zeros(n, dtype=np.float32)   #这里准备一个数组
    turn[1:] = np.abs(np.angle(np.exp(1j * (direction[1:, 0] - direction[:-1, 0])))).astype(np.float32)   #这里是把前一个的转向减去后一个的转向。exp是换出弧度制防止出现问题 
    on = np.zeros(n, dtype=np.float32)   #都在建立数组，这里是读取csv里面的数据
    off = np.zeros(n, dtype=np.float32)
    active = np.zeros(n, dtype=np.float32)
    duration = np.zeros(n, dtype=np.float32)
    names = defaultdict(list)   #创建词典
    for ob in obj_events:   #遍历每个物体的事件
        a = max(0.0, min(T, float(ob['a'])))   #开始时间
        b = max(a, min(T, float(ob['b'])))   #结束时间
        ia = min(n - 1, int(a / DT))     #转成时间桶下标
        ib = min(n - 1, int(b / DT))
        on[ia] += 1   #记录开始时间的数量
        off[ib] += 1   #记录结束时间的数量
        if b > a:
            active[ia:ib + 1] += 1  #计算这个时间桶里面有多少个事件
            duration[ia:ib + 1] += math.log1p(b - a)  #把覆盖当前桶的每个事件的完整时长做 ln(1+时长)，再相加。这样能够保留越长越大同时能压缩数值
        for idx in set((ia, ib)):  #记录名字
            if ob['name']:
                names[idx].append(ob['name'])
    active = active.clip(0, 12)
    idx = {int(t / DT) for t in np.arange(1.0, T - 0.1, 2.0)}
    idx.update((int(t / DT) for t in np.nonzero(on + off > 0)[0] if 0 < t * DT < T))   #记录事件发生的数组
    change = np.maximum(ch / 8, dv / 3)  #对元素进行缩放
    for i in range(2, n - 3):  #寻找运动变化的局部峰值
        if change[i] > 0.035 and change[i] >= change[i - 1] and (change[i] >= change[i + 1]) and (vv[i] > 0.1):
            idx.add(i)
    out = []
    for i in sorted(idx):
        t = (i + 0.5) * DT
        if not 0.2 < t < T - 0.2:
            continue
        local = hs[max(0, i - 2):min(n, i + 3)]   #取候选附近五个桶的手速
        steadiness = 1 / (1 + float(np.mean(local)) if len(local) else 1)   #1 / (1 + mean_speed)算出稳定度，值越高越慢
        h = float(max(hs[i], dv[i], ch[i] / 2))   #这里是当前的最强的信息动作
        o = float(on[i] + off[i])   #计算事件的边界数量
        o_near = max(0.0, min(1.0, float(on[max(0, i - 2):i + 3].sum() + off[max(0, i - 2):i + 3].sum())))   #附近有物体边界为1，没有为0
        F0 = float(hs[i])  # 左右手速度最大值，截断到4
        F1 = float(ch[i])  # 速度变化绝对值，截断到8（非加速度）
        F2 = float(vv[i])  # 左右手可见行比例
        F3 = float(turn[i])  # 仅左手方向差（弧度）
        F4 = float(speed[i, 0])  # 左手速度
        F5 = float(speed[i, 1])  # 右手速度
        F6 = float(min(5, on[i]))  # 物体事件起点数
        F7 = float(min(5, off[i]))  # 物体事件终点数
        F8 = float(active[i])  # 活跃物体事件数
        F9 = float(duration[i])  # 活动时长 log1p 求和
        F10 = float(min(h, o_near))  # 手部运动与物体事件边界共现
        F11 = float(max(0, hs[i] - hs[max(0, i - 2)]))  # 比约1秒前的手速正差
        F12 = float(max(0, hs[i] - hs[min(n - 1, i + 2)]))  # 比约1秒后的手速正差
        F13 = float(steadiness)  # 1/(1+局部平均手速)
        F14 = float(h > 0.1)  # 手部线索指示值
        F15 = float(o > 0)  # 物体事件边界指示值
        x = [F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15]
        flag_h = h > 0.1
        flag_o = o > 0
        source = 'JOINT' if flag_h and flag_o else 'HAND' if flag_h else 'OBJECT' if flag_o else 'FALLBACK'  #定义了手物的关系
        out.append({'t': round(t, 6), 'source': source, 'features': x,   
            'object_names': sorted(set(names.get(i, []))), 'cue_quality': round(max(h / 3, min(1, o)), 5),  #这里是线索强度分数，帮助两个候选尽可能分出区别出来
            'video_id': video_id})
    if len(out) < int(math.ceil(T / 30)):
        raise RuntimeError('TOO_FEW_PROPOSALS')
    if trace is not None:
        target = float(trace.get('requested_time_s', out[0]['t']))
        chosen = min(out, key=lambda c: (abs(c['t'] - target), c['t']))
        i = int(chosen['t'] / DT)
        a = max(0, i - 2)
        b = min(n, i + 3)
        raw_rows = []
        with gzip.open(hand_file, 'rt', encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                if a * DT <= val(r.get('time_s')) < b * DT:
                    raw_rows.append(r)
        trace.update(video_id=video_id, time_s=chosen['t'], bin_index=i, bin_start_s=i * DT,
            bin_end_s=(i + 1) * DT, F0_F15=dict(zip(FEATURES, chosen['features'])), candidate=chosen,
            raw_hand_rows=raw_rows,
            object_intervals=[o for o in obj_events if o['a'] <= b * DT and o['b'] >= a * DT],
            hand_file_sha256=sha256(hand_file), aggregation={key: arr[a:b].tolist() for key,
            arr in {'speed_by_side': speed, 'delta_by_side': delta, 'direction_by_side': direction,
            'counts_by_side': counts, 'visible_by_side': visible, 'hs': hs, 'ch': ch, 'dv': dv, 'vv': vv,
            'turn_left': turn, 'on': on, 'off': off, 'active': active, 'duration_log_sum': duration}.items()},
            aggregation_bin_indices=list(range(a, b)),
            boundary_proposal_rule='PRESERVED_ORIGINAL_SECOND_DIVISION_BY_DT',
            source_level='USER_SUPPLIED_HAND_CSV_AND_OBJECT_INTERVALS; NOT_RLE_REDECODE_PROOF')
    return (out, {'video_id': video_id, 'hand_valid_bins': coverage, 'proposals': len(out),  
        'hand': sum((c['source'] in ('HAND', 'JOINT') for c in out)),
        'object': sum((c['source'] in ('OBJECT', 'JOINT') for c in out)),
        'fallback': sum((c['source'] == 'FALLBACK' for c in out)), 'fixed_grid_interval_seconds': 2.0,
        'cue_bin_dt_seconds': DT})


def coarse_from_official_csv(vid, T, folder):
    path = Path(folder) / (vid[:3] + '_recipe_timestamps.csv')
    if not path.is_file():
        raise RuntimeError('STOP_NO_COARSE_TIMES_FOR_' + vid)
    out = []
    with path.open(encoding='utf-8-sig', newline='') as f:
        for i, r in enumerate(csv.DictReader(f)):
            if r.get('video_id') != vid:
                continue
            a = float(r['start_time'])
            b = T if r['end_time'].strip().lower() == 'end' else float(r['end_time'])
            if b - a > 0.01:
                out.append({'start_s': a, 'end_s': b,
                    'recipe_id': (r['recipe_id'] if r['recipe_id'].startswith(vid[:3] + '_') else vid[:3] + '_' + r['recipe_id']) if r.get('recipe_id') and r['recipe_id'] != '-' else None,
                    'coarse_text': r.get('high_level_activity_label') or '',
                    'origin': 'OFFICIAL_VIDEO_DERIVED_GT_ORACLE', 'idx': i})
    if not out:
        raise RuntimeError('STOP_EMPTY_COARSE_VIDEO ' + vid)
    return fill_gaps(out, T)

def fill_gaps(rows, T):
    for r in rows:
        if not 0 <= r['start_s'] <= r['end_s'] <= T + 0.1:
            raise RuntimeError('BAD_COARSE_TIMING')
    rows = sorted(rows, key=lambda r: (r['start_s'], r['end_s'], r['idx']))
    cur = 0.0
    extras = []
    for r in rows:
        if r['start_s'] > cur + 0.05:
            extras.append({'start_s': cur, 'end_s': r['start_s'], 'recipe_id': None,
                'coarse_text': 'UNKNOWN GAP', 'origin': 'UNLABELED_UNKNOWN', 'idx': -1 - len(extras)})
        cur = max(cur, r['end_s'])
    if T > cur + 0.05:
        extras.append({'start_s': cur, 'end_s': T, 'recipe_id': None, 'coarse_text': 'UNKNOWN GAP',
            'origin': 'UNLABELED_UNKNOWN', 'idx': -1 - len(extras)})
    return sorted(rows + extras, key=lambda r: (r['start_s'], r['end_s'], r['idx']))


def safe(values, bound: float=30.0) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    return np.clip(np.nan_to_num(a, nan=0.0, posinf=bound, neginf=-bound), -bound, bound)

def hand_object_score(c):   #手物综合分数
    x = c['features']
    return 0.17 * min(1, x[0] * 4) + 0.24 * min(1, x[1] * 1.3) + 0.22 * min(1,
        x[6] + x[7]) + 0.19 * min(1, x[8]) + 0.18 * min(1, x[14] + x[15])

def enrich(pool, objects, stages):
    out = []
    ts = np.array([c['t'] for c in pool])
    X = np.array([c['features'] for c in pool], float)
    if len(ts) == 0 or np.any(np.diff(ts) <= 0):
        raise Stop('CANDIDATE_TIMES_NOT_STRICTLY_SORTED')
    for i, c in enumerate(pool):
        active = [o for o in objects if o['a'] <= c['t'] <= o['b']]
        names = sorted(set(c.get('object_names', [])) | {o['name'] for o in active if o['name']})
        a = np.searchsorted(ts, c['t'] - 2)
        b = np.searchsorted(ts, c['t'] + 2, side='right')
        left = max(0, np.searchsorted(ts, c['t'] - 1, side='right') - 1)
        right = min(len(ts) - 1, np.searchsorted(ts, c['t'] + 1))
        current16, previous16, future16 = (X[i], X[left], X[right])
        local_mean16 = X[a:b].mean(axis=0)
        sensor64 = np.concatenate([current16, previous16, future16, local_mean16])
        sensor64 = np.sign(sensor64) * np.log1p(np.minimum(np.abs(sensor64), 30))
        stageids = [j for j, s in enumerate(stages) if s['start_s'] <= c['t'] < s['end_s']]
        out.append({**c, 'original_object_names': c.get('object_names', []), 'object_names': names,
            'object_tokens': sorted({t for s in names for t in words(s)}),
            'active_object_event_ids': sorted((o['event_id'] for o in active)), 'sensor': sensor64.tolist(),
            'stage_indices': stageids})
    return out

def frames(pool: list[dict]) -> np.ndarray:
    n = len(pool)
    if not n:
        raise ValueError('EMPTY_POOL')
    original16 = safe([c['features'][:16] for c in pool])
    sensor64 = safe([c['sensor'][:64] for c in pool])
    if original16.shape != (n, 16) or sensor64.shape != (n, 64):
        raise ValueError('FEATURE_DIMENSIONS_NOT_16_PLUS_64')
    previous16, future16, current16 = (sensor64[:, 16:32], sensor64[:, 32:48], sensor64[:, :16])
    times = np.asarray([c['t'] for c in pool], float)
    if n > 1 and (np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0)):
        raise ValueError('INVALID_SORTED_TIMES')
    context13 = np.empty((n, 13), float)  #最后13维的信息，提供空白的数组
    for i, c in enumerate(pool):
        prev = times[i] - times[i - 1] if i else 100.0
        nxt = times[i + 1] - times[i] if i + 1 < n else 100.0
        objects = c.get('object_tokens', [])
        events = c.get('active_object_event_ids', [])
        context13[i] = [float(hand_object_score(c)), float(c.get('cue_quality', 0.0)),  #这里填写了最后13维度的信息 :# 手物综合分数、线索强度。
            float(bool(objects)), float(min(5, len(objects))) / 5, float(bool(events)), float(min(5,  #是否有物体词元、物体词元数（截断至5后除以5）是否有活跃物体事件、活跃事件数（截断至5后除以5）
            len(events))) / 5, float(bool(c.get('stage_indices'))), float(c['source'] == 'HAND'),     #是否关联阶段、 # HAND/OBJECT/JOINT/FALLBACK四个来源指示值、
            float(c['source'] == 'OBJECT'), float(c['source'] == 'JOINT'), float(c['source'] == 'FALLBACK'),
            math.log1p(min(100, max(0, prev))), math.log1p(min(100, max(0, nxt)))]  #  距前/后候选的时间间隔（分别限制在0～100秒后取log1p）
    X141 = safe(np.column_stack((sensor64, original16, abs(current16 - previous16),
        abs(current16 - future16), abs(future16 - previous16), context13)))
    assert X141.shape == (n, 141)
    return X141


STEM = SnowballStemmer('english')

STOP = set('the a an and or to of in into on onto at with for from by then until is are be it this that your you some as while desired amount'.split())


@lru_cache(maxsize=50000)
def words(text):
    text = unicodedata.normalize('NFKC', str(text)).lower()
    return tuple(sorted({STEM.stem(w) for w in re.findall('[a-z]+', text) if len(w) > 1 and w not in STOP}))


# E. 原始数据生成候选（直接入口）
def validate_split(split: dict) -> tuple[dict, dict]:
    records = {row['video_id']: row for row in split['videos']}
    counts = Counter((row['split'] for row in split['videos']))
    if len(records) != 116 or counts != {'TRAIN': 70, 'DEV': 23, 'TEST': 23}:
        raise ValueError(f'Study protocol is not the agreed 70/23/23, got {counts}')
    groups = {}
    for entry in split['groups']:
        for vid in entry['video_ids']:
            if vid in groups:
                raise ValueError(f'Overlapping experiment groups: {vid}')
            groups[vid] = entry['group_id']
    if set(groups) != set(records):
        raise ValueError('Study groups do not cover 116 study videos')
    if len({groups[v] for v in records if records[v]['split'] == 'TRAIN'}) != 9:
        raise ValueError('TRAIN70 must use the existing nine study groups')
    group_scopes = defaultdict(set)
    for vid, group_id in groups.items():
        group_scopes[group_id].add(records[vid]['split'])
    overlaps = {group_id: sorted(scopes) for group_id, scopes in group_scopes.items() if len(scopes) != 1}
    if overlaps:
        raise ValueError(f'Study group crosses TRAIN/DEV/TEST: {overlaps}')
    return (records, groups)

def validate_official_download_receipt(download: dict, manifest: Path, split: Path,
    strict: bool=True) -> dict[str, str]:
    from data import BASE as BRISTOL_OFFICIAL, ONEDRIVE_SHARE, manifest_items
    source = download.get('source_type', 'bristol')
    source_url = download.get('dataset_base_url', '')
    expected_url = BRISTOL_OFFICIAL if source == 'bristol' else ONEDRIVE_SHARE
    if source not in ('bristol', 'onedrive') or source_url.rstrip('/') != expected_url.rstrip('/'):
        raise ValueError('Unrecognized official HD-EPIC raw source receipt')
    if not strict and source == 'bristol':
        return {row['file']: row['local_sha256'] for row in download['files']}
    if download.get('file_count') != 233 or len(download.get('files', [])) != 233:
        raise ValueError('Official download report is incomplete (expected 233 files)')
    required = dict(manifest_items(manifest, split))
    reported = {row['file']: row for row in download['files']}
    if set(reported) != set(required) or any((row.get('official_md5') != required[name] or row.get('matches_official_md5') is not True or (not isinstance(row.get('local_sha256'),
        str)) or (len(row['local_sha256']) != 64) for name, row in reported.items())):
        raise ValueError('Original-file receipt does not match the official 233-file MD5 manifest')
    return {name: row['local_sha256'] for name, row in reported.items()}

def raw_command() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hd', type=Path, required=True,
        help='Official HD-EPIC raw directory (Bristol or author OneDrive)')
    p.add_argument('--annotations', type=Path, required=True, help='Official GitHub annotations git checkout')
    p.add_argument('--split', type=Path, required=True, help='Researcher-defined 116-video protocol')
    p.add_argument('--download-report', type=Path, required=True, help='New official download SHA256 receipt')
    p.add_argument('--setup-receipt', type=Path,
        help='reports/SETUP_INPUTS.json; verify the exact official Git checkouts and split')
    p.add_argument('--out', type=Path, required=True, help='Brand-new interim directory')
    p.add_argument('--video', help='Only process this video for a one-video review')
    p.add_argument('--trace-video', default='P02-20240209-184316')
    p.add_argument('--trace-time', type=float, default=872.25)
    a = p.parse_args()
    split = read_json(a.split)
    records, _ = validate_split(split)
    download = read_json(a.download_report)
    reference_md5 = a.annotations.parent / 'hd-epic-downloader' / 'data' / 'md5.txt'
    downloaded_hashes = validate_official_download_receipt(download, reference_md5, a.split,
        strict=a.setup_receipt is not None)
    if a.setup_receipt:
        from data import verify_setup_sources
        md5_file = a.annotations.parent / 'hd-epic-downloader' / 'data' / 'md5.txt'
        proof = verify_setup_sources(md5_file, a.split, a.setup_receipt)
        if download.get('source_setup_receipt_sha256') != proof['setup_receipt_sha256']:
            raise ValueError('Raw download report is not linked to this official source setup')
        if download.get('study_split_sha256') != sha256(a.split):
            raise ValueError('Raw download report and current frozen split differ')
        official_annotations = proof['repositories']['hd-epic-annotations']['local_directory']
        if str(a.annotations.resolve()) != official_annotations:
            raise ValueError('Annotations folder differs from verified official Git checkout')
    chosen = [a.video] if a.video else sorted(records)
    if not set(chosen) <= set(records):
        raise ValueError('Video not in study protocol')
    if a.out.exists():
        raise FileExistsError(f'Keep previous run; pass a new --out directory: {a.out}')
    archive = a.hd / 'Hands-Masks' / 'contours_preds.zip'
    assoc = a.annotations / 'scene-and-object-movements' / 'assoc_info.json'
    masks = a.annotations / 'scene-and-object-movements' / 'mask_info.json'
    activities = a.annotations / 'high-level' / 'activities'
    source_hashes = {'archive': sha256(archive), 'assoc': sha256(assoc), 'masks': sha256(masks),
        'study_split': sha256(a.split), 'download_report': sha256(a.download_report)}
    if downloaded_hashes['Hands-Masks/contours_preds.zip'] != source_hashes['archive']:
        raise ValueError('RLE archive differs from verified official download')
    with zipfile.ZipFile(archive) as source_zip:
        for vid in chosen:
            if 'contours_preds/' + vid + '.json' not in source_zip.namelist():
                raise ValueError(f'Official RLE video member missing: {vid}')
    source_objects = read_json(assoc)
    source_masks = read_json(masks)
    missing_assoc_ids = sorted(set(records) - set(source_objects))
    missing_mask_ids = sorted(set(records) - set(source_masks))
    print(f'[OBJECT SOURCE] missing_assoc_video_keys={len(missing_assoc_ids)} missing_mask_video_keys={len(missing_mask_ids)}',
        flush=True)
    subset_assoc = {vid: source_objects.get(vid, {}) for vid in records}
    subset_masks = {vid: source_masks.get(vid, {}) for vid in records}
    events, anchors, object_report = normalize_official_objects(subset_assoc, subset_masks,  #对物体进行分类
        source_hashes['assoc'], source_hashes['masks'])
    obj_by_video = defaultdict(list)
    for event in events:
        obj_by_video[event['video_id']].append({'a': event['start_s'], 'b': event['end_s'],
            'name': event['object_name_normalized'], 'event_id': event['event_id'], 'oracle': True})
    a.out.mkdir(parents=True)
    (a.out / 'objects').mkdir()
    write_jsonl_new(a.out / 'objects/events.jsonl', events)
    write_jsonl_new(a.out / 'objects/anchors.jsonl', anchors)
    write_json_new(a.out / 'objects/source_receipt.json', {'inputs': source_hashes,
        'tracks': object_report['n_tracks'], 'anchors': object_report['n_sparse_anchors'],
        'missing_assoc_video_keys': missing_assoc_ids, 'missing_mask_video_keys': missing_mask_ids,
        'source_paths': {'assoc_info': str(assoc.resolve()), 'mask_info': str(masks.resolve()),
        'hand_rle_zip': str(archive.resolve())}})
    stage_map, durations, counts = ({}, {}, {})
    for i, vid in enumerate(chosen, 1):
        times = a.hd / 'Videos' / vid[:3] / (vid + '_mp4_to_vrs_time_ns.csv')
        log = a.hd / 'Videos' / vid[:3] / (vid + '_vrs_to_mp4_log.json')
        for sidecar in (times, log):
            relative = sidecar.relative_to(a.hd).as_posix()
            if sha256(sidecar) != downloaded_hashes[relative]:
                raise ValueError(f'Official download sidecar changed: {relative}')
        duration_s = int(read_json(log)['video_duration_ns']) / 1000000000.0
        durations[vid] = duration_s
        try:
            stages = coarse_from_official_csv(vid, duration_s, activities)
            coarse_source = 'OFFICIAL_HIGH_LEVEL_ACTIVITIES'
        except RuntimeError as exc:
            if records[vid]['split'] != 'TRAIN' or not str(exc).startswith(('STOP_NO_COARSE_TIMES_FOR_',
                'STOP_EMPTY_COARSE_VIDEO')):
                raise
            print('[TRAIN UNAVAILABLE COARSE]', vid, str(exc), flush=True)
            stages = []
            coarse_source = 'TRAIN_EMPTY_AS_ORIGINAL_METHOD'
        stage_map[vid] = stages
        outdir = a.out / 'videos' / vid
        outdir.mkdir(parents=True)
        hand_path = outdir / 'hand.csv.gz'
        decoded = decode_hand_video(archive, times, log, vid, hand_path)  #解码手部信息
        trace = {'requested_time_s': a.trace_time} if vid == a.trace_video else None
        v3, statistics = make_candidates(vid, hand_path, obj_by_video[vid], duration_s, trace)   #计算16维信息
        pool = enrich(v3, obj_by_video[vid], stages)  #丰富成64维信息
        output = outdir / 'candidates.json.gz'
        write_json_new(output, {'video_id': vid, 'pool': pool, 'duration_s': duration_s,  #这里是输出的信息
            'stages': stages, 'statistics': statistics})
        if trace is not None:
            index = next((i for i, row in enumerate(pool) if row['t'] == trace['time_s']))
            trace.update(candidate_index=index, enriched_candidate=pool[index],
                HGB_141=frames(pool)[index].tolist(), hand_file_sha256=sha256(hand_path),   #141维信息
                object_source_sha256=source_hashes['assoc'], archive_sha256=source_hashes['archive'])
            write_json_new(outdir / 'trace.json', trace)
        activity_csv = activities / (vid[:3] + '_recipe_timestamps.csv')
        write_json_new(outdir / 'receipt.json', {'video_id': vid,  #计算完信息开始输出
            'origin': 'OFFICIAL_DOWNLOAD_NO_LOCAL_FROZEN_INPUT', 'inputs': {'archive': source_hashes['archive'],
            'sidecar': sha256(times), 'clock_log': sha256(log), 'objects': source_hashes['assoc'],
            'object_masks': source_hashes['masks'],
            'activity_csv': sha256(activity_csv) if activity_csv.is_file() else None,
            'split': source_hashes['study_split']}, 'input_paths': {'hand_rle_archive': str(archive.resolve()),
            'hand_rle_member': f'contours_preds/{vid}.json', 'clock_csv': str(times.resolve()),
            'clock_log': str(log.resolve()), 'object_associations': str(assoc.resolve()),
            'object_mask_anchors': str(masks.resolve()), 'coarse_activity_csv': str(activity_csv.resolve())},
            'output_paths': {'hand_30_columns': str(hand_path.resolve()),
            'candidates_F0_F15_and_sensor64': str(output.resolve())}, 'coarse_source': coarse_source,
            'object_source_video_key_present': vid in source_objects,
            'object_masks_video_key_present': vid in source_masks, 'duration_s': duration_s,
            'frames': decoded['clock']['row_count'], 'hand_rows': decoded['hand_rows'],
            'candidate_count': len(pool), 'outputs': {'hand_csv_sha256': sha256(hand_path),
            'candidate_sha256': sha256(output)}})
        counts[vid] = {'frames': decoded['clock']['row_count'], 'hand_rows': decoded['hand_rows'],
            'candidates': len(pool)}
        print(f"[RAW {i}/{len(chosen)}] {vid} frames={counts[vid]['frames']} hand={counts[vid]['hand_rows']} V3={len(pool)}",
            flush=True)
    write_json_new(a.out / 'stages.json', stage_map)
    write_json_new(a.out / 'durations.json', durations)
    write_json_new(a.out / 'raw_status.json', {'status': 'OFFICIAL_RAW_CANDIDATES_GENERATED',
        'videos': len(chosen), 'full_116': len(chosen) == 116, 'official_sources': source_hashes,
        'paths': {'official_raw': str(a.hd.resolve()), 'official_annotations': str(a.annotations.resolve()),
        'download_report': str(a.download_report.resolve()), 'split': str(a.split.resolve()),
        'generated_candidates_root': str(a.out.resolve())},
        'object_source_missing_video_keys': {'assoc': missing_assoc_ids, 'masks': missing_mask_ids},
        'video_counts': counts,
        'note': 'Uncorrected historical V3 object boundary conversion; numeric parity with old run not asserted.'})
    print(f"[DONE] {len(chosen)} new-RAW videos; {a.out / 'raw_status.json'}", flush=True)


# F. TRAIN/DEV/TEST 输入打包（直接入口）
def official_sop(recipes: dict) -> dict:
    sop = {}
    for recipe_id, recipe in recipes.items():
        steps = recipe['steps']
        sop[recipe_id] = [{'recipe_id': recipe_id, 'ordinal': i, 'step_id': step_id, 'text': text,
            'object_roles': [], 'provenance': 'OFFICIAL_AS_PERFORMED_RECIPE_STEPS',
            'pre_execution_independence': 'UNKNOWN'} for i, (step_id, text) in enumerate(steps.items(), 1)]
    return sop

def official_train_fine(narrations: pd.DataFrame, train_ids: set[str]) -> dict:
    columns = {'video_id', 'start_timestamp', 'end_timestamp', 'unique_narration_id', 'verbs',
        'nouns', 'main_actions'}
    if not columns <= set(narrations.columns):
        raise ValueError(f'Official narrations missing {sorted(columns - set(narrations.columns))}')
    labels: dict[str, list] = {}
    for row in narrations.itertuples(index=False):
        vid = str(row.video_id)
        if vid not in train_ids:
            continue
        start, end = (float(row.start_timestamp), float(row.end_timestamp))
        if not 0 <= start < end:
            raise ValueError(f'Invalid official TRAIN action interval: {vid} {start} {end}')
        labels.setdefault(vid, []).append({'a': start, 'b': end, 'uid': str(row.unique_narration_id),
            'verbs': list(row.verbs), 'nouns': list(row.nouns),
            'main_actions': [list(pair) for pair in row.main_actions]})
    return labels

def inputs_command() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True, type=Path,
        help='New official raw candidate output, data/interim/raw')
    p.add_argument('--annotations', required=True, type=Path, help='Official GitHub checkout')
    p.add_argument('--setup-receipt', type=Path,
        help='reports/SETUP_INPUTS.json: verify official Git checkout before reading labels')
    p.add_argument('--split', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path, help='New processed output directory')
    a = p.parse_args()
    protocol = read_json(a.split)
    records, groups = validate_split(protocol)
    if a.setup_receipt:
        from data import verify_setup_sources
        md5_file = a.annotations.parent / 'hd-epic-downloader' / 'data' / 'md5.txt'
        proof = verify_setup_sources(md5_file, a.split, a.setup_receipt)
        if str(a.annotations.resolve()) != proof['repositories']['hd-epic-annotations']['local_directory']:
            raise ValueError('Official annotation directory differs from source setup receipt')
    status = read_json(a.raw / 'raw_status.json')
    if a.setup_receipt and status.get('official_sources', {}).get('study_split') != sha256(a.split):
        raise ValueError('Candidate generation and machine learning use different research splits')
    if not status['full_116'] or status['videos'] != 116:
        raise ValueError('Prepare all 116 videos from official RAW before making ML inputs')
    durations = read_json(a.raw / 'durations.json')
    stages = read_json(a.raw / 'stages.json')
    if set(durations) != set(records) or set(stages) != set(records):
        raise ValueError('Raw candidate video set differs from study split')
    pools = {}
    for vid in sorted(records):
        pools[vid] = read_json(a.raw / 'videos' / vid / 'candidates.json.gz')['pool']
    steps_file = a.annotations / 'high-level/complete_recipes.json'
    sop = official_sop(read_json(steps_file))
    fine_file = a.annotations / 'narrations-and-action-segments/HD_EPIC_Narrations.pkl'
    official = pd.read_pickle(fine_file)
    train_ids = {v for v, r in records.items() if r['split'] == 'TRAIN'}
    fine = official_train_fine(official, train_ids)
    a.out.mkdir(parents=True, exist_ok=False)
    generated_files = {}
    for scope in ('TRAIN', 'DEV', 'TEST'):
        vids = sorted((v for v in records if records[v]['split'] == scope))
        doc = {'schema': 'HGKS_PORTABLE_INPUT_V1', 'scope': scope,
            'pools': {v: pools[v] for v in vids}, 'durations': {v: durations[v] for v in vids},
            'provenance': {'candidate_origin': 'OFFICIAL_HD_EPIC_RLE_NEW_COMPUTATION',
            'raw_status_sha256': sha256(a.raw / 'raw_status.json'), 'study_split_sha256': sha256(a.split),
            'recipe_steps_sha256': sha256(steps_file),
            'official_narrations_sha256': sha256(fine_file) if scope == 'TRAIN' else None,
            'independence_note': 'As-performed recipes; pre-execution independence UNKNOWN'}}
        if scope == 'TRAIN':
            doc['fines'] = fine
            doc['groups'] = {v: groups[v] for v in vids}
        else:
            doc['stages'] = {v: stages[v] for v in vids}
            doc['sop'] = sop
        output = a.out / (scope + str(len(vids)) + '.json.gz')
        write_json_new(output, doc)
        generated_files[scope] = str(output.resolve())
        count = sum((len(pools[v]) for v in vids))
        print(f'[ML INPUT {scope}] videos={len(vids)} candidates={count} file={output}', flush=True)
    write_json_new(a.out / 'input_receipt.json', {'source_raw': str(a.raw.resolve()),
        'source_raw_status_sha256': sha256(a.raw / 'raw_status.json'),
        'source_official_annotations': str(a.annotations.resolve()),
        'source_recipe_steps_path': str(steps_file.resolve()),
        'source_official_narrations_path': str(fine_file.resolve()),
        'source_split_path': str(a.split.resolve()), 'actual_generated_files': generated_files,
        'study_split_sha256': sha256(a.split), 'recipe_steps_sha256': sha256(steps_file),
        'official_narrations_sha256': sha256(fine_file), 'TRAIN_Fine_labeled_video_count': len(fine),
        'fine_is_used_only_for_training': True, 'DEV_TEST_gold_in_inputs': False,
        'SOP_pre_execution_independence': 'UNKNOWN'})


def main():
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ('raw', 'inputs'):
        raise SystemExit('用法: python prepare.py raw|inputs --help')
    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    if command == 'raw':
        raw_command()
    else:
        inputs_command()

if __name__ == '__main__':
    main()
