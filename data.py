"""只处理底层数据：读写、安全检查、官方 RLE 时间和 Object 标注。
从这里导出的函数由 prepare.py 使用；不进行机器学习。"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, io, json, math, statistics, zipfile
import bisect, re, subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np


# 1. JSON 和输入校验
SCHEMA = 'HGKS_PORTABLE_INPUT_V1'

INFERENCE_FORBIDDEN = {'fine', 'fines', 'fine_action', 'fine_actions', 'qa', 'qa_gold', 'targets',
    'ground_truth', 'test_labels'}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def write_json_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    if path.suffix == '.gz':
        with path.open('xb') as f:
            with gzip.GzipFile(filename='', fileobj=f, mode='wb', mtime=0) as z:
                z.write(payload)
    else:
        with path.open('xb') as f:
            f.write(payload)
    return sha256(path)

def _validate_pool(vid, pool):
    if not isinstance(pool, list) or len(pool) < 1:
        raise ValueError('HGKS_POOL_MISSING ' + vid)
    times = np.asarray([c['t'] for c in pool], float)
    if not np.all(np.isfinite(times)) or (np.diff(times) <= 0).any():
        raise ValueError('HGKS_POOL_NOT_SORTED ' + vid)
    for i, c in enumerate(pool):
        if len(c['features']) < 16 or len(c['sensor']) < 64 or (not all((k in c for k in ('source',
            'object_tokens', 'object_names', 'active_object_event_ids', 'stage_indices')))):
            raise ValueError(f'HGKS_BAD_CANDIDATE_SCHEMA {vid}:{i}')
        if len(c['features']) == 16 and (not np.all(np.isfinite(c['features']))):
            raise ValueError('HGKS_FEATURE_NONFINITE')

def load_train(path, strict70=True):
    doc = read_json(path)
    if doc.get('schema') != SCHEMA or doc.get('scope') != 'TRAIN':
        raise ValueError('HGKS_EXPECT_TRAIN_ONLY_FILE')
    pools = doc['pools']
    fines = doc['fines']
    groups = doc['groups']
    durations = doc['durations']
    if set(pools) != set(groups) or set(pools) != set(durations) or (not set(fines) <= set(pools)):
        raise ValueError('HGKS_TRAIN_FILE_COHERENCE')
    if strict70 and (len(pools) != 70 or len(set(groups.values())) != 9):
        raise ValueError('HGKS_EXPECT_TRAIN70_9_GROUPS')
    for vid, pool in pools.items():
        _validate_pool(vid, pool)
    return (pools, fines, groups, durations, doc.get('provenance', {}))

def reject_inference_gold(value):
    if isinstance(value, dict):
        if INFERENCE_FORBIDDEN & set(value):
            raise ValueError('HGKS_TARGET_ANNOTATION_LEAKAGE_IN_INFERENCE_INPUT')
        for child in value.values():
            reject_inference_gold(child)
    elif isinstance(value, list):
        for child in value:
            reject_inference_gold(child)

def load_inference(path, expected_split=None):
    doc = read_json(path)
    if doc.get('schema') != SCHEMA or doc.get('scope') not in ('DEV', 'TEST'):
        raise ValueError('HGKS_EXPECT_UNLABELED_DEV_OR_TEST_FILE')
    if expected_split and doc['scope'] != expected_split:
        raise ValueError('HGKS_REQUESTED_INFERENCE_SPLIT_MISMATCH')
    if INFERENCE_FORBIDDEN & set(doc):
        raise ValueError('HGKS_TARGET_ANNOTATION_LEAKAGE_IN_INFERENCE_INPUT')
    expected = {'schema', 'scope', 'pools', 'durations', 'stages', 'sop', 'provenance'}
    if set(doc) - expected:
        raise ValueError('HGKS_UNRECOGNIZED_INFERENCE_METADATA_FIELDS')
    reject_inference_gold(doc)
    pools = doc['pools']
    durations = doc['durations']
    stages = doc['stages']
    sop = doc['sop']
    if set(pools) != set(durations) or set(pools) != set(stages):
        raise ValueError('HGKS_INFERENCE_FILE_COHERENCE')
    for vid, pool in pools.items():
        _validate_pool(vid, pool)
        for c in pool:
            if set(c) & INFERENCE_FORBIDDEN:
                raise ValueError('HGKS_ANNOTATION_LEAKAGE_IN_CANDIDATE')
    return (pools, durations, stages, sop, doc.get('provenance', {}), doc['scope'])

class Stop(RuntimeError):
    pass

def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()

def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]

# 2. 已下载官方原件的来源核验（不包含下载器）
BASE = 'https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/'

MASK = 'Hands-Masks/contours_preds.zip'

SUFFIXES = ('_mp4_to_vrs_time_ns.csv', '_vrs_to_mp4_log.json')

ONEDRIVE_SHARE = 'https://uob-my.sharepoint.com/:f:/g/personal/wq23021_bristol_ac_uk/IgA0PLwURLmfQZOl_J256iZtAY_uzFr0gTXVMWCWY4MSGd0?e=6C6wNa'

def file_hash(path: Path, method: str) -> str:
    digest = hashlib.new(method)
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def manifest_items(manifest: Path, split: Path) -> list[tuple[str, str]]:
    experiment = json.loads(split.read_text(encoding='utf-8-sig'))
    video_ids = {row['video_id'] for row in experiment['videos']}
    counts = {scope: sum((row['split'] == scope for row in experiment['videos'])) for scope in ('TRAIN',
        'DEV', 'TEST')}
    if len(video_ids) != 116 or counts != {'TRAIN': 70, 'DEV': 23, 'TEST': 23}:
        raise ValueError('study/split.json must define TRAIN70/DEV23/TEST23')
    selected = {}
    for text in manifest.read_text(encoding='utf-8').splitlines():
        if not text.strip():
            continue
        official_md5, relative = text.split(maxsplit=1)
        relative = relative.removeprefix('./')
        filename = Path(relative).name
        needed_time_file = relative.startswith('Videos/') and filename.endswith(SUFFIXES) and (filename.removesuffix('_mp4_to_vrs_time_ns.csv').removesuffix('_vrs_to_mp4_log.json') in video_ids)
        if relative == MASK or needed_time_file:
            if relative in selected:
                raise ValueError('Duplicate source entry: ' + relative)
            if len(official_md5) != 32 or any((c not in '0123456789abcdefABCDEF' for c in official_md5)):
                raise ValueError('Invalid official MD5: ' + relative)
            selected[relative] = official_md5.lower()
    if len(selected) != 233 or MASK not in selected:
        raise ValueError(f'Expected 233 files (1+116×2), got {len(selected)}')
    return sorted(selected.items())

def verify_setup_sources(manifest: Path, split: Path, setup_receipt: Path) -> dict:
    receipt = json.loads(setup_receipt.read_text(encoding='utf-8'))
    expected_manifest = Path(receipt['official_md5_manifest'])
    expected_split = Path(receipt['study_split']['file'])
    if manifest.resolve() != expected_manifest.resolve():
        raise ValueError(f'MD5 manifest path differs from setup receipt: {manifest}')
    if split.resolve() != expected_split.resolve():
        raise ValueError(f'Study split path differs from setup receipt: {split}')
    if file_hash(manifest, 'sha256') != receipt['official_md5_manifest_sha256']:
        raise ValueError('Official GitHub md5.txt SHA256 changed since setup')
    if file_hash(split, 'sha256') != receipt['study_split']['sha256']:
        raise ValueError('Frozen 116-video research split SHA256 changed since setup')
    observed = {}
    for name, entry in receipt['official_repositories'].items():
        local = Path(entry['local_directory'])
        remote = subprocess.check_output(['git', '-C', str(local), 'remote', 'get-url', 'origin'],
            text=True).strip()
        expected_remote = entry['official_git_url']
        if remote.removesuffix('.git').rstrip('/') != expected_remote.removesuffix('.git').rstrip('/'):
            raise ValueError(f'Git source URL changed for {name}: {remote}')
        if expected_remote.removesuffix('.git').rstrip('/') != 'https://github.com/hd-epic/' + name:
            raise ValueError(f'Nonofficial Git source for {name}: {expected_remote}')
        commit = subprocess.check_output(['git', '-C', str(local), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != entry['commit']:
            raise ValueError(f'Official Git source commit changed for {name}: {commit}')
        dirty = subprocess.check_output(['git', '-C', str(local), 'status', '--porcelain'], text=True).strip()
        if dirty:
            raise ValueError(f'Official Git checkout modified locally: {local}')
        observed[name] = {'git_url': remote, 'commit': commit, 'local_directory': str(local.resolve())}
    print('[OFFICIAL MD5 SOURCE]', observed['hd-epic-downloader']['git_url'], flush=True)
    print('[OFFICIAL MD5 GIT COMMIT]', observed['hd-epic-downloader']['commit'], flush=True)
    print('[OFFICIAL MD5 SHA256]', file_hash(manifest, 'sha256'), flush=True)
    print('[RESEARCH SPLIT ORIGIN]', receipt['study_split']['copied_from'], flush=True)
    print('[RESEARCH SPLIT SHA256]', file_hash(split, 'sha256'), flush=True)
    print('[OFFICIAL ANNOTATIONS GIT COMMIT]', observed['hd-epic-annotations']['commit'], flush=True)
    print('[SOURCE PROVENANCE VERIFIED]', setup_receipt.resolve(), flush=True)
    return {'setup_receipt_sha256': file_hash(setup_receipt, 'sha256'), 'repositories': observed,
        'study_split_origin': receipt['study_split']['copied_from'], 'study_split_sha256': file_hash(split,
        'sha256')}

# 3. 手部 RLE → 每帧左右手 CSV
SIDECAR_COLUMNS = {'mp4_time_ns', 'relative_vrs_device_time_ns', 'vrs_device_time_ns'}

HAND_FIELDS = ['video_id', 'frame_index', 'time_s', 'hand', 'visible', 'center_x_norm',
    'center_y_norm', 'bbox_x1_norm', 'bbox_y1_norm', 'bbox_x2_norm', 'bbox_y2_norm', 'area_pixels',
    'area_ratio', 'image_width', 'image_height', 'velocity_x_norm_per_s', 'velocity_y_norm_per_s',
    'speed_norm_per_s', 'direction_rad', 'speed_change_norm_per_s', 'motion_window_s', 'validity',
    'missing_reason', 'confidence', 'source_quality', 'mask_tier', 'side_verified_by_rgb',
    'clock_verified', 'sidecar_sha256', 'source_csv_sha256']

def finite_number(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None

def decode_counts(encoded: str | list[int]) -> list[int]:
    if isinstance(encoded, list):
        if any((type(n) is not int or n < 0 for n in encoded)):
            raise ValueError('RLE runs must be nonnegative integers')
        return encoded
    if not isinstance(encoded, str):
        raise ValueError('RLE counts must be a string or integer list')
    runs: list[int] = []
    index = 0
    while index < len(encoded):
        value = shift = 0
        while True:
            if index >= len(encoded):
                raise ValueError('Truncated COCO RLE count')
            code = ord(encoded[index]) - 48
            index += 1
            if not 0 <= code <= 63:
                raise ValueError('Invalid COCO RLE character')
            value |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    value |= -1 << shift
                break
            if shift > 60:
                raise ValueError('Unreasonably large RLE count')
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            raise ValueError('Negative decoded RLE run')
        runs.append(value)
    return runs

def rle_geometry(rle: dict[str, Any]) -> dict[str, Any]:
    if set(rle) != {'size', 'counts'}:
        raise ValueError('Unexpected RLE fields')
    height, width = rle['size']
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise ValueError('Invalid mask dimensions')
    runs = decode_counts(rle['counts'])
    if sum(runs) != height * width:
        raise ValueError('RLE runs do not span the declared image')
    position = area = sum_x = sum_y = 0
    x1, y1, x2, y2 = (width, height, -1, -1)
    for run_index, count in enumerate(runs):
        if run_index % 2 and count:
            start_x, start_y = divmod(position, height)
            end_x, end_y = divmod(position + count - 1, height)
            area += count
            x1, x2 = (min(x1, start_x), max(x2, end_x))
            if start_x == end_x:
                sum_x += start_x * count
                sum_y += (start_y + end_y) * count // 2
                y1, y2 = (min(y1, start_y), max(y2, end_y))
            else:
                first_count, last_count = (height - start_y, end_y + 1)
                middle_count = end_x - start_x - 1
                sum_x += start_x * first_count + end_x * last_count
                sum_x += (start_x + end_x) * middle_count * height // 2
                sum_y += (start_y + height - 1) * first_count // 2
                sum_y += end_y * last_count // 2
                sum_y += middle_count * height * (height - 1) // 2
                y1, y2 = (0, height - 1)
        position += count
    return {'height': height, 'width': width, 'area': area, 'area_fraction': area / (height * width),
        'visible': bool(area), 'centroid_x': sum_x / area if area else None,
        'centroid_y': sum_y / area if area else None, 'bbox_x1': x1 if area else None,
        'bbox_y1': y1 if area else None, 'bbox_x2': x2 if area else None, 'bbox_y2': y2 if area else None}

def decode_counts_vectorized(encoded: str | list[int]):  #对COCO RLE解码
    import numpy as np
    if isinstance(encoded, list):
        return np.asarray(decode_counts(encoded), dtype=np.int64)
    if not isinstance(encoded, str):
        raise ValueError('RLE counts must be a string or integer list')
    if not encoded:
        return np.zeros(0, dtype=np.int64)
    try:
        codes = np.frombuffer(encoded.encode('ascii'), dtype=np.uint8).astype(np.int64) - 48
    except UnicodeEncodeError as exc:
        raise ValueError('Invalid COCO RLE character') from exc
    if np.any((codes < 0) | (codes > 63)):
        raise ValueError('Invalid COCO RLE character')
    ends = np.flatnonzero(codes & 32 == 0)
    if not len(ends) or ends[-1] != len(codes) - 1:
        raise ValueError('Truncated COCO RLE count')
    starts = np.r_[0, ends[:-1] + 1]
    lengths = ends - starts + 1
    if int(lengths.max()) > 10:
        return np.asarray(decode_counts(encoded), dtype=np.int64)
    shifts = 5 * (np.arange(len(codes)) - np.repeat(starts, lengths))
    runs = np.add.reduceat(np.left_shift(codes & 31, shifts), starts)
    signed = codes[ends] & 16 != 0
    runs[signed] -= np.left_shift(np.int64(1), 5 * lengths[signed])
    runs[1::2] = np.cumsum(runs[1::2])
    runs[2::2] = np.cumsum(runs[2::2])
    if np.any(runs < 0):
        raise ValueError('Negative decoded RLE run')
    return runs

def rle_geometry_vectorized(rle: dict[str, Any]) -> dict[str, Any]:  #解码手部的掩码
    import numpy as np
    if set(rle) != {'size', 'counts'}:
        raise ValueError('Unexpected RLE fields')
    height, width = rle['size']
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise ValueError('Invalid mask dimensions')
    if height * width * (height + width) > 2 ** 62:
        return rle_geometry(rle)
    runs = decode_counts_vectorized(rle['counts'])  #对COCO RLE进行解码
    if bool(np.any(runs > height * width)) or int(runs.sum()) != height * width:
        raise ValueError('RLE runs do not span the declared image')
    ends = np.cumsum(runs)     #结束 
    foreground_counts = runs[1::2]  #属于手部的像素 
    nonempty = foreground_counts > 0
    end = ends[1::2][nonempty]
    start = end - foreground_counts[nonempty]
    area = int(foreground_counts.sum())   #计算手部面积
    if not area:
        return {'height': height, 'width': width, 'area': 0, 'area_fraction': 0.0, 'visible': False,
            'centroid_x': None, 'centroid_y': None, 'bbox_x1': None, 'bbox_y1': None, 'bbox_x2': None,
            'bbox_y2': None}
    start_x, start_y = np.divmod(start, height)
    end_prefix_x, end_prefix_y = np.divmod(end, height)
    end_x, end_y = np.divmod(end - 1, height)  #下面两部在计算手部的质心。核心是用前景像素的横纵坐标总和除以面积得到的
    sum_x = int((height * end_prefix_x * (end_prefix_x - 1) // 2 + end_prefix_x * end_prefix_y - height * start_x * (start_x - 1) // 2 - start_x * start_y).sum())
    sum_y = int((end_prefix_x * height * (height - 1) // 2 + end_prefix_y * (end_prefix_y - 1) // 2 - start_x * height * (height - 1) // 2 - start_y * (start_y - 1) // 2).sum())
    crosses_column = bool(np.any(start_x != end_x))
    return {'height': height, 'width': width, 'area': area, 'area_fraction': area / (height * width),
        'visible': True, 'centroid_x': sum_x / area, 'centroid_y': sum_y / area,
        'bbox_x1': int(start_x.min()), 'bbox_y1': 0 if crosses_column else int(start_y.min()),
        'bbox_x2': int(end_x.max()), 'bbox_y2': height - 1 if crosses_column else int(end_y.max())}

class TimebaseError(ValueError):
    pass

def _hash(path: Path, algorithm: str='sha256') -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TimebaseError(message)

@dataclass(frozen=True)
class FrameTimebase:
    video_id: str
    mp4_time_s: tuple[float, ...]
    relative_vrs_s: tuple[float, ...]
    vrs_device_ns: tuple[int, ...]
    index_base: int
    audit: dict

    def time_for_frame(self, frame_index: int, *, video_id: str | None=None) -> float:
        _require(video_id is None or video_id == self.video_id, 'VIDEO_NAMESPACE_MISMATCH')
        _require(type(frame_index) is int, 'FRAME_INDEX_MUST_BE_INTEGER')
        ordinal = frame_index - self.index_base
        _require(0 <= ordinal < len(self.mp4_time_s), 'FRAME_INDEX_OUT_OF_RANGE')
        return self.mp4_time_s[ordinal]

def load_timebase(csv_path: Path, log_path: Path, video_id: str, frame_indices: Iterable[int], #读取时间信息
    index_base: int | None=None) -> FrameTimebase:
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        _require(set(reader.fieldnames or []) == SIDECAR_COLUMNS, 'SIDECAR_SCHEMA_MISMATCH')
        rows = list(reader)
    log = _json(log_path)
    _require(len(rows) >= 2, 'SIDECAR_TOO_SHORT')
    _require(type(log.get('num_mp4_frames')) is int and len(rows) == log['num_mp4_frames'],
        'MP4_FRAME_COUNT_MISMATCH')
    mp4_ns = [float(row['mp4_time_ns']) for row in rows]
    relative_ns = [int(row['relative_vrs_device_time_ns']) for row in rows]
    device_ns = [int(row['vrs_device_time_ns']) for row in rows]
    _require(all((math.isfinite(t) for t in mp4_ns)), 'NONFINITE_MP4_TIME')
    _require(mp4_ns[0] == 0.0, 'MP4_TIME_ORIGIN_NOT_ZERO')
    intervals = [b - a for a, b in zip(mp4_ns, mp4_ns[1:])]
    _require(all((dt > 0 for dt in intervals)), 'MP4_TIME_RESET_OR_DUPLICATE_SEGMENT')
    origin = int(log['first_video_timestamp_ns'])
    _require(all((abs_time - relative == origin for abs_time, relative in zip(device_ns,
        relative_ns))), 'VRS_OFFSET_INCONSISTENT_WITH_LOG')
    _require(all((b >= a for a, b in zip(device_ns, device_ns[1:]))), 'VRS_TIME_RESET_OR_MULTISEGMENT')
    log_duration_ns = int(log['video_duration_ns'])
    _require(int(log['end_video_timestamp_ns']) - origin == log_duration_ns, 'LOG_DURATION_OFFSET_MISMATCH')
    nominal = statistics.median(intervals)
    _require(mp4_ns[-1] <= log_duration_ns <= mp4_ns[-1] + 2 * nominal,
        'SIDECAR_CLOCK_EXTENT_INCONSISTENT_WITH_LOG')
    raw_indices = list(frame_indices)
    _require(raw_indices and all((type(index) is int for index in raw_indices)),
        'MASK_INDICES_MUST_BE_NONEMPTY_INTEGERS')
    indices = sorted(set(raw_indices))
    possible = [base for base in (0, 1) if all((0 <= index - base < len(rows) for index in indices))]
    if index_base is None:
        _require(len(possible) == 1, 'AMBIGUOUS_MASK_INDEX_BASE: explicit non-GT origin evidence required')
        index_base = possible[0]
    _require(index_base in (0, 1) and index_base in possible, 'MASK_INDEX_BASE_OR_RANGE_MISMATCH')
    residual = [(v - m) / 1000000.0 for v, m in zip(relative_ns, mp4_ns)]
    xs = [v / 1000000000.0 for v in mp4_ns]  #转换时间的代码，源代码提供了逐帧的时间映射
    mean_x, mean_y = (statistics.mean(xs), statistics.mean(residual))
    denominator = sum(((x - mean_x) ** 2 for x in xs))
    trend_ms_per_min = 60 * sum(((x - mean_x) * (y - mean_y) for x, y in zip(xs, residual))) / denominator
    device_intervals = [b - a for a, b in zip(device_ns, device_ns[1:])]
    maximum_30hz_error_ns = max((abs(value - i * 1000000000.0 / 30) for i, value in enumerate(mp4_ns)))
    audit = {'video_id': video_id, 'numeric_status': 'PASS',
        'mapping': 'CSV_ROW_ORDINAL_TO_MP4_TIME_NS', 'row_count': len(rows), 'mp4_index_base': 0,
        'mask_index_base': index_base, 'mask_frame_count': len(indices), 'mask_frame_min': indices[0],
        'mask_frame_max': indices[-1], 'mask_index_coverage': len(indices) / len(rows),
        'missing_mask_frame_keys': len(rows) - len(indices), 'effective_mask_index_rate': 1.0,
        'missing_mp4_rows_vs_log': log['num_mp4_frames'] - len(rows),
        'duplicate_mp4_timestamp_rows': len(mp4_ns) - len(set(mp4_ns)),
        'dense_mask_indices_equal_mp4_sequence': indices == list(range(index_base, len(rows) + index_base)),
        'mp4_start_s': xs[0], 'mp4_last_frame_s': xs[-1], 'log_duration_s': log_duration_ns / 1000000000.0,
        'median_mp4_interval_s': nominal / 1000000000.0,
        'inferred_fps_from_explicit_timestamps': 1000000000.0 / nominal,
        'minimum_mp4_interval_s': min(intervals) / 1000000000.0,
        'maximum_mp4_interval_s': max(intervals) / 1000000000.0,
        'constant_30hz_within_one_ns': maximum_30hz_error_ns <= 1.0,
        'maximum_30hz_formula_error_ns': maximum_30hz_error_ns, 'vrs_origin_ns': origin,
        'vrs_offset_identity_verified': True, 'device_minus_mp4_first_ms': residual[0],
        'device_minus_mp4_last_ms': residual[-1], 'device_minus_mp4_min_ms': min(residual),
        'device_minus_mp4_max_ms': max(residual),
        'device_minus_mp4_linear_trend_ms_per_min': trend_ms_per_min,
        'trend_caveat': 'Resampling correspondence residual; not calibrated physical clock drift',
        'vrs_duplicate_timestamp_intervals': sum((dt == 0 for dt in device_intervals)),
        'vrs_large_timestamp_intervals': sum((dt > 1.5 * nominal for dt in device_intervals)),
        'log_num_skipped_frames': log.get('num_skipped_frames'),
        'log_num_duplicated_frames': log.get('num_duplicated_frames_'),
        'log_num_invalid_frames': log.get('num_invalid_frames_'),
        'log_down_sampling_factor': log.get('down_sampling_factor_'),
        'counter_caveat': 'Conversion counters and timestamp-repeat counts have distinct semantics; neither renumbers MP4 rows',
        'csv_sha256': _hash(csv_path), 'log_sha256': _hash(log_path), 'gt_used_for_clock': False,
        'rgb_pts_visual_verified': False}
    return FrameTimebase(video_id, tuple(xs), tuple((t / 1000000000.0 for t in relative_ns)),
        tuple(device_ns), index_base, audit)

def normalized_geometry(geom):
    visible = geom['visible'] in (True, 'True', 'true', '1', 1)
    w, h = (finite_number(geom.get('width')), finite_number(geom.get('height')))
    if not visible:
        return {'visible': False, 'x': None, 'y': None, 'box': [None] * 4,
            'area': finite_number(geom.get('area')), 'area_ratio': finite_number(geom.get('area_fraction')),
            'width': w, 'height': h}
    if w is None or h is None or w <= 0 or (h <= 0):
        raise ValueError('VISIBLE_HAND_WITHOUT_IMAGE_DIMENSIONS')
    x = finite_number(geom['centroid_x']) / max(w - 1, 1)
    y = finite_number(geom['centroid_y']) / max(h - 1, 1)
    box = [finite_number(geom[k]) / max(d - 1, 1) for k, d in zip(('bbox_x1', 'bbox_y1', 'bbox_x2',
        'bbox_y2'), (w, h, w, h))]
    area = finite_number(geom['area'])
    fraction = finite_number(geom['area_fraction'])
    if not (0 <= x <= 1 and 0 <= y <= 1 and (0 <= box[0] <= box[2] <= 1) and (0 <= box[1] <= box[3] <= 1) and (area > 0) and (0 < fraction <= 1)):
        raise ValueError('INVALID_VISIBLE_HAND_GEOMETRY')
    if not math.isclose(fraction, area / (w * h), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('HAND_AREA_RATIO_MISMATCH')
    return {'visible': True, 'x': x, 'y': y, 'box': box, 'area': area, 'area_ratio': fraction,
        'width': w, 'height': h}

def hand_rows(data, clock, video, source_sha):  #这个函数在计算速度
    previous = {}
    for frame in sorted(map(int, data)):   #frame 是帧号，t 是该帧对应的真实秒数，side 是左手或者右手。
        t = clock.time_for_frame(frame)
        for side in ('left', 'right'):
            mask = data[str(frame)].get(side)
            raw = rle_geometry_vectorized(mask) if mask else {'visible': False, 'width': None,  #这里返回了手的横纵坐标
                'height': None, 'area': None, 'area_fraction': None}
            geom = normalized_geometry(raw)  #这里是归一化，把横纵坐标像素除以总像素，得到归一化的坐标
            vx = vy = speed = direction = speed_change = window = None
            if geom['visible']:
                old = previous.get(side)  #前一帧的信息，这里0是编号，1是时间，2是x坐标，3是y坐标
                if old and old[0] == frame - 1:
                    window = t - old[1]
                    vx, vy = ((geom['x'] - old[2]) / window, (geom['y'] - old[3]) / window)   #计算速度，vx是横向速度，vy是纵向速度
                    speed = math.hypot(vx, vy)      # 计算速度的大小，平方开根
                    direction = math.atan2(vy, vx) if speed else None  # 计算速度的方向
                    speed_change = speed - old[4] if old[4] is not None else None
                previous[side] = (frame, t, geom['x'], geom['y'], speed)
            else:
                previous.pop(side, None)
            values = [video, frame, t, side, geom['visible'], geom['x'], geom['y'], *geom['box'],
                geom['area'], geom['area_ratio'], geom['width'], geom['height'], vx, vy, speed, direction,
                speed_change, window, 'OK' if geom['visible'] else 'MISSING',
                None if geom['visible'] else 'NO_PREDICTED_MASK_OCCLUSION_OFFSCREEN_OR_MODEL_FAILURE_UNKNOWN', None,
                'SIDECAR_MP4_INDEX_VERIFIED_RGB_UNVERIFIED', 'contours_preds', False, True,
                clock.audit['csv_sha256'], source_sha]
            assert len(values) == len(HAND_FIELDS)
            yield dict(zip(HAND_FIELDS, values))  #这里会输出一个hand.csv.gz

def decode_hand_video(archive_path, csv_path, log_path, video, out_path):
    out_path = Path(out_path)
    if out_path.exists():
        raise ValueError('REFUSING_TO_OVERWRITE ' + str(out_path))
    if Path(csv_path).name != video + '_mp4_to_vrs_time_ns.csv' or Path(log_path).name != video + '_vrs_to_mp4_log.json':
        raise ValueError('SIDECAR_VIDEO_NAMESPACE_MISMATCH') 
    with zipfile.ZipFile(archive_path) as archive:  #读取手部的掩码信息
        info = archive.getinfo('contours_preds/' + video + '.json')
        content = archive.read(info)
    source_sha = hashlib.sha256(content).hexdigest()
    data = json.loads(content)
    if not isinstance(data, dict) or any((not key.isdigit() or key != str(int(key)) for key in data)):
        raise ValueError('RAW_MASK_FRAME_KEY_INVALID')
    if any((not isinstance(h, dict) or set(h) - {'left', 'right'} for h in data.values())):
        raise ValueError('RAW_MASK_SIDE_OR_CONTAINER_INVALID')
    shapes = {tuple(mask['size']) for sides in data.values() for mask in sides.values() if mask}
    if len(shapes) > 1:
        raise ValueError('INCONSISTENT_HAND_IMAGE_DIMENSIONS')
    frames = sorted(map(int, data))
    clock = load_timebase(csv_path, log_path, video, frames)  #读取时间信息
    if clock.index_base != 0 or not clock.audit['dense_mask_indices_equal_mp4_sequence']:
        raise ValueError('DENSE_ZERO_BASED_MASK_MP4_NAMESPACE_NOT_PROVEN')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    observed = []
    visible = {'left': 0, 'right': 0}
    motion = {'left': 0, 'right': 0}
    count = 0
    traced_motion = None
    with out_path.open('xb') as binary, gzip.GzipFile(filename='', mode='wb', fileobj=binary, mtime=0,
        compresslevel=1) as zipped, io.TextIOWrapper(zipped, encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=HAND_FIELDS)
        writer.writeheader()
        for row in hand_rows(data, clock, video, source_sha):
            writer.writerow(row)
            count += 1
            visible[row['hand']] += int(row['visible'])
            motion[row['hand']] += int(row['speed_norm_per_s'] is not None)
            if len(observed) < 8:
                observed.append(row)
            if traced_motion is None and row['speed_norm_per_s'] is not None:
                frame = row['frame_index']
                side = row['hand']
                traced_motion = {'hand_row': row, 'source_json_pointer': f'/{frame}/{side}',
                    'raw_rle': data[str(frame)][side], 'pixel_geometry': rle_geometry_vectorized(data[str(frame)][side]),
                    'previous_pixel_geometry': rle_geometry_vectorized(data[str(frame - 1)][side]),
                    'previous_time_s': clock.time_for_frame(frame - 1), 'sidecar_data_row_0based': frame}
    return {'status': 'RLE_REDECODE_EXECUTED', 'video_id': video, 'clock': clock.audit,
        'hand_rows': count, 'visible_rows': visible, 'motion_valid_rows': motion,
        'source_member': info.filename, 'source_json_sha256': source_sha, 'source_member_crc32': info.CRC,
        'output_sha256': sha256(out_path), 'example_rows': observed, 'motion_trace': traced_motion,
        'geometry_source': 'HISTORICAL_RLE_GEOMETRY_AND_HANDS_MECHANISM',
        'csv_header_status': 'HISTORICAL_CORE_HAND_FIELDS_30_COLUMNS', 'hand_fields': HAND_FIELDS,
        'source_archive': str(Path(archive_path)), 'source_sidecar': str(Path(csv_path)),
        'source_conversion_log': str(Path(log_path)), 'rgb_pts_visual_verified': False,
        'SAM2_run_by_this_program': False, 'contact_truth': False}

# 4. 官方 Object → 时间事件及稀疏锚点
def normalize_name(value):
    return ' '.join(re.findall('[a-z0-9]+', value.lower()))

def box_valid(box):
    return isinstance(box, list) and len(box) == 4 and all((type(v) in (int,
        float) and math.isfinite(v) for v in box)) and (0 <= box[0] <= box[2]) and (0 <= box[1] <= box[3])

def normalize_official_objects(assoc, mask_info, assoc_sha=None, mask_sha=None):
    if not isinstance(assoc, dict) or not isinstance(mask_info, dict):
        raise ValueError('BOTH_OFFICIAL_OBJECT_JSON_OBJECTS_REQUIRED')
    events, keys, identifiers = ([], set(), set())
    for video, associations in assoc.items():
        for association_id, association in associations.items():
            name = association['name']
            if not isinstance(name, str):
                raise ValueError('OFFICIAL_OBJECT_NAME_NOT_TEXT')
            for track in association['tracks']:
                a, b = track['time_segment']
                if not (type(a) in (int, float) and type(b) in (int,
                    float) and math.isfinite(a) and math.isfinite(b) and (0 <= a <= b)):
                    raise ValueError('OFFICIAL_OBJECT_INTERVAL_INVALID')
                event_id = f"{video}/{association_id}/{track['track_id']}"
                if event_id in identifiers:
                    raise ValueError('DUPLICATE_OFFICIAL_OBJECT_EVENT_ID ' + event_id)
                identifiers.add(event_id)
                mask_ids = track['masks']
                linked = []
                for mid in mask_ids:
                    keys.add((video, mid))
                    mask = mask_info.get(video, {}).get(mid)
                    if mask is not None:
                        frame = mask['frame_number']
                        if type(frame) is not int or frame < 0:
                            raise ValueError('OBJECT_MASK_FRAME_INDEX_INVALID')
                        linked.append(mask)
                linked.sort(key=lambda mask: mask['frame_number'])
                first, last = (linked[0], linked[-1]) if linked else ({}, {})
                events.append({'video_id': video, 'association_id': association_id,
                    'track_id': track['track_id'], 'object_name': name, 'start_s': a, 'end_s': b,
                    'start_frame': first.get('frame_number'), 'end_frame': last.get('frame_number'),
                    'start_bbox_xyxy': first.get('bbox'), 'end_bbox_xyxy': last.get('bbox'),
                    'start_fixture': first.get('fixture'), 'end_fixture': last.get('fixture'),
                    'sparse_mask_ids': mask_ids, 'oracle_annotation': True, 'source_sha256': assoc_sha,
                    'event_id': event_id, 'object_name_normalized': normalize_name(name),
                    'observation_tier': 'ORACLE_OBJECT_EVENT', 'continuous_trajectory': False,
                    'true_contact_verified': False,
                    'anchor_rule': 'Earliest/latest linked mask frame; not guaranteed exact event boundary',
                    'geometry_units': 'Official MP4 pixel xyxy; no interpolation',
                    'mask_info_sha256': mask_sha})
    anchors, missing, invalid = ([], [], [])
    for video, mid in sorted(keys):
        raw = mask_info.get(video, {}).get(mid)
        if raw is None:
            missing.append({'video_id': video, 'mask_id': mid,
                'reason': 'OFFICIAL_ASSOC_MASK_REFERENCE_NOT_FOUND'})
            continue
        box = raw.get('bbox')
        if not box_valid(box):
            invalid.append({'video_id': video, 'mask_id': mid,
                'reason': 'OFFICIAL_BOX_NEGATIVE_OR_INVALID_RETAINED_NO_CLIPPING'})
        anchors.append({'video_id': video, 'mask_id': mid, 'frame_index': raw['frame_number'],
            'bbox_xyxy_pixels': box, 'scene_location_xyz': raw.get('3d_location'),
            'scene_coordinate_units': 'OFFICIAL_RECONSTRUCTION_UNITS_REFERENCE_NOT_INDEPENDENTLY_VERIFIED',
            'fixture_id': raw.get('fixture'), 'oracle_annotation': True, 'source_sha256': mask_sha,
            'continuous_trajectory': False, 'bbox_geometry_valid': box_valid(box)})
    report = {'status': 'NORMALIZED_PROVIDED_OBJECT_JSON', 'n_tracks': len(events),
        'n_sparse_anchors': len(anchors), 'missing_mask_references': missing,
        'invalid_boxes_retained': invalid, 'mask_geometry_used_in_F0_F15': False,
        'normalization_basis': 'HISTORICAL_CORE_AND_FREEZE_PREPARE',
        'full_V1_event_record_parity': 'UNVERIFIED_V1_SOURCE_RECORDS_NOT_PROVIDED'}
    return (events, anchors, report)

def write_jsonl_new(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

