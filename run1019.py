"""1019线性执行入口：新训练→新预测→全部选择→冻结→等价检查→原版评价。"""
# 从这里开始读整个工程。PowerShell 中可以复制下面命令；反引号表示本条命令还没结束。
# 已核验 paths.json 指向的输入、两套 Python、CLIP 模型和参考文件存在。
# MY_CACHED_ANNOTATED_001 在本轮检查时不存在；以后重复运行须更换 --out，不能覆盖已有结果。
# Set-Location C:\Project08\1019
# .\.venv\Scripts\python.exe -X utf8 run1019.py run `
#   --out results\MY_CACHED_ANNOTATED_001 `
#   --mode cached `
#   --reference-selections _reference\EXPECTED_1019_SELECTIONS.json `
#   --reference-results _reference\EXPECTED_1019_RESULTS.csv
# python.exe 是 Lite 训练/预测/评价使用的本地解释器；-X utf8 指定文本读写默认编码。
# run1019.py 是实际入口文件；run 表示训练到评价的全流程；--out 是本次新实验的目录。
# cached 读取已经核验的真实 RGB 特征缓存，不表示复制历史选择或历史评价 CSV。
# 两份 reference 只在新选点冻结后、新评价完成后分别参与核验，不提供给选点函数。
# 按 Enter 后先执行本文件的模块级语句：导入模块、设置环境变量、建立常量、定义函数。
# 导入也可能执行被导入模块的顶层语句，不能把 main 称为解释器执行的第一个 Python 函数。
# 等到最下方 if __name__ == '__main__' 成立才调用 main；这是本文件的业务流程入口。
# 本轮仅补注释，没有执行这条完整 DEV23 命令；已冻结的 cached/Raw 状态见阅读说明。
from __future__ import annotations
import os
# 子进程继承这些环境变量：模型只用本地文件，CPU 数值库使用历史的 4 个线程。
# 4 是当前源码固定参数，优化来源未核实；注释任务不改变线程数或依赖版本。
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0',
                  OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUTF8='1')
import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from data import load_inference, read_json, sha256, write_json_new
from prepare import budget
import baselines

ROOT = Path(__file__).resolve().parent
# ROOT 是本文件所在的 1019 绝对目录，避免从其他终端目录启动时读错输入。
# range 右端不包含：20261012 到 20261021 共 10 个基础种子，属于两个 Random 候选域。
# 同一种方法换种子重复 10 次，不是增加 10 种不同算法；实际视频种子还会加视频名字符和。
SEEDS = range(20261012, 20261022)
# 冻结结果中的输入权限说明。Lite 使用 Hand/Object/coarse Oracle 及 TRAIN Fine 监督；
# 时间方法、HSV、CLIP、SOP 各自拥有不同信息，不能把同 DEV23/同 K 称为严格同输入比较。
INPUT_INFORMATION = {
    '1017_LITE_FINAL': 'HAND_RLE+OBJECT_ORACLE+COARSE_ORACLE+TRAIN_FINE_SUPERVISION',
    'UNIFORM_FULL_ALL_FRAMES': 'OFFICIAL_ALL_FRAME_TIME',
    'UNIFORM_FULL_VIDEO': 'TIME_4K',
    'UNIFORM_POOL_CURRENT': 'LITE_CANDIDATE_POOL_TIME',
    'INVERSE_EXP_SOP_STAGE': 'TIME_4K+COARSE_ORACLE',
    'VSUMM_HIST_KMEANS': 'RGB_HSV40',
    'MAXVOL_HIST_DOPT': 'RGB_HSV40',
    'MAXINFO_CLIP_MAXVOL_ADAPT': 'RGB_CLIP512',
    'MAXINFO_CHUNK_CLIP_ADAPT': 'RGB_CLIP512',
    'AKS_SOP_CLIP_RECONSTRUCTED_QUERY_ADAPT': 'RGB_CLIP512+SOP',
}


def load_paths(file):
    """paths.json相对路径→1019内路径；只有现成的工具解释器允许外部路径。"""
    # main 调用；file 是 --paths 指定的 Path，默认 ROOT/paths.json。
    # read_json 在 data.py 中读取 JSON 字典：键如 dev/features，值是文件或目录路径字符串。
    # 字典推导式逐项把相对路径接到 ROOT 后 resolve 为绝对 Path；原本绝对的工具路径保持绝对。
    # 返回 paths: dict[str, Path] 到 main，随后传给 choose_frames / run_native_evaluation。
    paths = {k: (ROOT / v).resolve() for k, v in read_json(file).items()}
    paths['root'] = ROOT
    return paths


def command(args, out, label):
    """实际执行CLI→stdout/stderr/命令/退出码；失败即停止，不伪造成功回执。"""
    # main / run_native_evaluation 调用。args 是 [解释器, Python 文件, 参数...]，out 是实验 Path，
    # label 是日志文件前缀；把所有 Path 转成 str 后才能统一传给操作系统启动子进程。
    # Ctrl+单击 command 进入此处；重点看 args 中实际 Python 文件，再打开那个文件的 main。
    # 本函数不在当前调用栈里训练模型；subprocess.run 启动另一个进程，并等待它退出。
    args = list(map(str, args))
    logs = Path(out) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    # cwd 固定为 1019；capture_output 保存输出流；UTF-8 解码供人查看，不改模型的计算。
    # 返回 proc 是 CompletedProcess，包含 returncode/stdout/stderr，不是模型或预测数组。
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    (logs / (label + '.stdout.txt')).write_text(proc.stdout, 'utf-8')
    (logs / (label + '.stderr.txt')).write_text(proc.stderr, 'utf-8')
    write_json_new(logs / (label + '.command.json'), {
        'args': args, 'returncode': proc.returncode, 'elapsed_s': time.monotonic()-start,
        'cwd': str(ROOT), 'created_utc': datetime.now(timezone.utc).isoformat()})
    if proc.returncode:
        raise RuntimeError(label + ': ' + proc.stderr[-2500:])
    # 正常结束时返回 None；调用者下一步从子进程实际写出的模型/JSON读取数据。
    # -2500 是错误信息截断长度，不是算法参数；monotonic 秒差记录实际运行耗时。
    print('CLI_PASS', label, flush=True)


def load_protocol(paths, out, video=None):
    """无标签DEV输入/split/时钟→冻结协议；不打开官方Fine、历史选择和历史分数。"""
    # choose_frames 调用。paths 来自 load_paths；out 是本次实验目录；video 是可选视频 ID。
    # Ctrl+单击 load_inference 到 data.py：读取 data/portable/DEV23.json.gz 并检查推理权限。
    # pools[v] 是候选列表（P08 有 1110 个）；durations[v] 是秒；stages[v] 是 coarse 阶段；
    # sop 是 recipe_id→SOP 步骤文本；provenance/split 是来源和划分信息，不是 Fine 标签。
    pools, durations, stages, sop, provenance, split = load_inference(paths['dev'], 'DEV')
    # split.json 的列表推导式只取 DEV 视频，再排序固定处理顺序；不读取或处理 TEST23 输入。
    # 23 视频、25995 候选、预算和 932 都是已核验的 DEV23 协议量，不是搜索得到的超参数。
    ids = sorted(r['video_id'] for r in read_json(paths['split'])['videos'] if r['split'] == 'DEV')
    if set(ids) != set(pools) or len(ids) != 23 or sum(map(len, pools.values())) != 25995:
        raise ValueError('DEV23_INPUT_CHANGED')
    K = {v: budget(durations[v]) for v in ids}
    # Ctrl+单击 budget 到 prepare.py 看继承的时长→预算规则；P08 duration=1184.51197917 秒时 K=40。
    if sum(K.values()) != 932:
        raise ValueError('DEV23_K_CHANGED')
    if video:
    # 先验证完整输入，才缩小本次选择范围；这没有把单视频的评价协议改成完整 DEV23。
        if video not in ids:
            raise ValueError('VIDEO_NOT_IN_DEV')
        ids = [video]
    clock_sha = {}
    for v in ids:
    # cp 是官方 MP4 帧→VRS 时钟 CSV，lp 是相应转换 log；v[:3] 如 P08 是参与者子目录。
    # FEATURE_RECEIPT.json 保存复制进 1019 时的来源 SHA256，逐个核对防止换错时钟文件。
        cp = paths['clocks'] / v[:3] / (v + '_mp4_to_vrs_time_ns.csv')
        lp = cp.with_name(v + '_vrs_to_mp4_log.json')
        feature_receipt = read_json(paths['features'] / v / 'FEATURE_RECEIPT.json')
        if sha256(cp) != feature_receipt['video_asset']['CSV_sha256'] or sha256(lp) != feature_receipt['video_asset']['log_sha256']:
            raise ValueError('OFFICIAL_CLOCK_CHANGED')
        clock_sha[v] = {'csv_sha256': sha256(cp), 'log_sha256': sha256(lp)}
    protocol = {'split': split, 'video_ids': ids, 'K': {v: K[v] for v in ids},
                'K_total': sum(K[v] for v in ids), 'input_sha256': sha256(paths['dev']),
                'split_sha256': sha256(paths['split']), 'clocks': clock_sha,
                'Fine_access': False, 'historical_results_access': False, 'grid_ratio': 4,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'source_sha256': {p.name: sha256(p) for p in ROOT.glob('*.py')}}
    # protocol 记录本轮真实输入、每视频 K、时钟和当前源码 SHA；grid_ratio=4 是历史 4K 规则。
    # 只补注释也会改变当前源码 SHA；不能据此断言旧冻结时间点改变，更不能重写旧清单。
    # write_json_new 在 data.py 拒绝覆盖已有文件；返回五元组给 choose_frames 的同名变量。
    write_json_new(Path(out) / 'protocol.json', protocol)
    return pools, durations, stages, sop, protocol


def choose_frames(paths, out, mode='cached', video=None):
    """逐视频：真缓存/真RGB→原Baseline函数；Lite读取本轮predict.py的真实输出。"""
    # main 的 select 分支调用；完整 run 会由另一套 feature_python 启动该分支。
    # 参数路径来自 paths.json；mode 是 cached/raw；video=None 时选择所有 23 个 DEV 视频。
    # 数据流：load_protocol→grid→get_features→all_indices→snap_to_mp4→freeze_selections。
    # 本函数不返回数组给 main（正常返回 None），而把所有方法的真实选点冻结到 out。
    # 外部方法历史依赖与Lite不同；本分支由paths中的feature_python运行。
    from vision import get_features, snap_to_mp4
    import sklearn
    if sklearn.__version__ != '1.9.1' or np.__version__ != '2.4.0':
        raise ValueError('BASELINE_ENVIRONMENT_DIFFERS_FROM_FROZEN_RUN')
    pools, durations, stages, sop, protocol = load_protocol(paths, out, video)
    # Alt+左箭头返回这里：protocol['video_ids'] 决定循环视频，其他四项仍来自同一 DEV23 输入。
    write_json_new(Path(out) / 'baseline_environment.json', {
        'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__,
        'sklearn': sklearn.__version__, 'platform': platform.platform(), 'mode': mode})
    lite = read_json(Path(out) / 'lite_raw/DEV_LITE_FROZEN.json')
    # lite 必须是本轮 predict.py 输出；同时核对 DEV 输入 SHA 和本轮新训练模型 SHA。
    # select 不负责训练/预测，所以单独调试 select 必须先在 out 下准备好 train、lite_raw。
    if lite['input_sha256'] != protocol['input_sha256']:
        raise ValueError('LITE_INPUT_CHANGED')
    model = Path(out) / 'train/HGKS_LITE_MODEL.joblib'
    if lite['model_sha256'] != sha256(model):
        raise ValueError('LITE_MODEL_CHANGED')
    methods = {}
    for v in protocol['video_ids']:
        K = budget(durations[v])
        cp = paths['clocks'] / v[:3] / (v + '_mp4_to_vrs_time_ns.csv')
        clock = np.loadtxt(cp, delimiter=',', skiprows=1, usecols=0, ndmin=1) / 1e9
    # CSV 第一列是纳秒，除 1e9 转为秒；clock:(真实帧数,) float，索引就是零起始 MP4 帧号。
    # np.diff 检查相邻帧时钟严格递增；它是后面 searchsorted 能正确找帧的前提。
        log = read_json(cp.with_name(v + '_vrs_to_mp4_log.json'))
        if len(clock) != log['num_mp4_frames'] or not np.all(np.diff(clock) > 0):
            raise ValueError('INVALID_OFFICIAL_CLOCK')
        times = baselines.grid(durations[v], K)
    # Ctrl+单击 grid：4K 个等时长区段的中点；P08 times:(160,)，单位秒。
    # dict.fromkeys 按首次出现顺序对 recipe_id 去重，再按该顺序展开 SOP 文本，不能用无序 set。
        recipe_ids = list(dict.fromkeys(s['recipe_id'] for s in stages[v] if s.get('recipe_id')))
        steps = [s for rid in recipe_ids for s in sop[rid]]
        h, X, q = get_features(v, times, steps, clock, paths, out, mode)
    # Ctrl+单击 get_features 到 vision.py：cached 读真实 NPZ；raw 才解码 JPEG、重算 HSV/CLIP。
    # 返回 h:(160,40) HSV，X:(160,512) 帧 CLIP，q:(512,) SOP；这里 X 不是训练的 X141。
    # Alt+左箭头回此处，看下一行将三个数组传给 all_indices，K=40，不改变 Lite 的 141 维。
        found = baselines.all_indices(v, times, K, stages[v], h, X, q)
    # Ctrl+单击 all_indices 到 baselines.py，再逐个进 VSUMM/MaxVol/MaxInfo/Chunk/AKS。
    # found 是 方法名→K 个整数下标 的字典；下标属于 times，不是原 MP4 帧号。
        found['AKS_SOP_CLIP_RECONSTRUCTED_QUERY_ADAPT'] = found.pop('AKS_SOP_CLIP_ADAPT')
    # 上行只把同一个索引数组换成正式报告名。下面 times[ix] 用整数数组同时取出 K 个请求秒数。
    # RANDOM_K 不进入正式表；它由下方两个候选域、各 10 个真实种子结果替代。
        requests = {m: times[ix].tolist() for m, ix in found.items() if m != 'RANDOM_K'}
        requests['UNIFORM_FULL_ALL_FRAMES'] = clock[baselines.uniform_indices(len(clock), K)].tolist()
        for seed in SEEDS:
            for domain, arr in [('4K', times), ('ALL_FRAMES', clock)]:
                ix = baselines.random_indices(len(arr), K, seed + sum(map(ord, v)))
    # ord 将视频名每个字符转整数；同视频/同基础种子总是得到同一实际种子，且不依赖 Python hash。
    # 4K 用 160 个网格点，ALL_FRAMES 用所有官方帧；random_indices 无放回抽 K 个下标。
                requests[f'RANDOM_{domain}_SEED_{seed}'] = arr[ix].tolist()
        pool_times = np.array([c['t'] for c in pools[v]])
    # UNIFORM_POOL_CURRENT 使用 Lite 候选池（P08 为 1110），不能拿 160 点 RGB 网格替代它。
        requests['UNIFORM_POOL_CURRENT'] = pool_times[baselines.uniform_pool_indices(pool_times, K)].tolist()
        requests['1017_LITE_FINAL'] = [r['time_s'] for r in lite['predictions'][v]['selected']]
        for method, ts in requests.items():
            mapped = snap_to_mp4(sorted(map(float, ts)), clock)
    # Ctrl+单击 snap_to_mp4 到 vision.py：请求秒数排序→首个不早于请求的官方帧→消除帧碰撞。
    # P08 UNIFORM_FULL_VIDEO 网格下标2：请求18.50799967453125秒，实际帧556、18.53333333333333秒。
    # mapped 是字典，保留请求秒、真实帧号、实际秒和误差；评价使用实际秒 selected_times_s。
            indices = mapped['official_frame_indices_zero_based']
            if len(indices) != K or len(set(indices)) != K or any(i < 0 or i >= len(clock) for i in indices):
                raise ValueError('INVALID_PHYSICAL_FRAME_SELECTION')
            row = {'K': K, **mapped, 'clock_csv_sha256': sha256(cp), 'video_id': v}
            if method in found:
                row['selected_grid_indices'] = list(map(int, found[method]))
    # 外部网格方法保存原索引便于回查；methods.setdefault(method,{}) 得到该方法的视频字典。
    # 这是 method→video_id→row 的两层数据表，不是另一层算法转发。
            methods.setdefault(method, {})[v] = row
        print('REAL_SELECTION', v, 'K', K, 'methods', len(requests), flush=True)
    freeze_selections(methods, protocol, paths, out, mode)
    # Ctrl+单击看下面：全部视频选择完成才冻结，此前没有打开 DEV Fine 或历史参考选择。


def freeze_selections(methods, protocol, paths, out, mode):
    """30套新选择→每方法JSON/原评价传输格式/回执→总SHA清单；此时仍未读Gold。"""
    # choose_frames 调用，methods 是上方累积的三层字典；protocol 是本轮输入及代码的来源回执。
    # 30=10 套非随机方法（含 Lite）+2 个 Random 候选域×10 个种子；--video 仍是30套但各1视频。
    # 每方法生成 frozen_selections/<method>.json 和 native/<method>/DEV_METHOD_FROZEN.json；
    # 最后总清单写 frozen_manifest.json，正常返回 None，控制权回到 choose_frames 再到 main。
    if len(methods) != 30:
        raise ValueError('REQUIRE_30_SELECTION_FILES')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'Fine_access': False,
                'historical_results_access': False, 'mode': mode, 'methods': {}}
    for method, videos in sorted(methods.items()):
        frozen = Path(out) / 'frozen_selections' / (method + '.json')
        info = INPUT_INFORMATION.get(method, 'TIME_4K' if method.startswith('RANDOM_4K_') else 'OFFICIAL_ALL_FRAME_TIME')
    # identity 是本工程的比较身份：INTERNAL_CONTROL 内部对照，NEW_ADAPTATION 本地适配。
    # MaxInfo 是本地 D-opt/Logdet 适配，AKS 是 SOP 查询适配，不声称官方完整算法复现。
        identity = 'INTERNAL_CONTROL' if method in ('1017_LITE_FINAL', 'UNIFORM_FULL_ALL_FRAMES',
                    'UNIFORM_POOL_CURRENT') or method.startswith('RANDOM_ALL_FRAMES_') else 'NEW_ADAPTATION'
        write_json_new(frozen, {'method_id': method, 'identity': identity, 'split': 'DEV',
                               'n_videos': len(videos), 'videos': videos,
                               'input_sha256': protocol['input_sha256'], 'Fine_access': False,
                               'input_information': info, 'selector_version': baselines.VERSION,
                               'source_sha256': protocol['source_sha256'],
                               'model_sha256': sha256(Path(out) / 'train/HGKS_LITE_MODEL.joblib') if method == '1017_LITE_FINAL' else '',
                               'CLIP_model_id': 'openai/clip-vit-base-patch32' if 'CLIP512' in info else ''})
        transport = Path(out) / 'native' / method / 'DEV_METHOD_FROZEN.json'
        predictions = {v: {'K': r['K'], 'selected': [{'time_s': t} for t in r['selected_times_s']]}
                       for v, r in videos.items()}
    # 双层推导式只转换输出格式：每视频 K 及实际 MP4 时间列表，适配原 evaluate.py 的输入结构。
    # source_selection_sha256 把评价传输文件连到选点文件；receipt 再记录传输文件 SHA。
        write_json_new(transport, {'schema': 'HGKS_FINAL_PREDICTION_V1', 'split': 'DEV',
                                  'predictions': predictions, 'input_sha256': protocol['input_sha256'],
                                  'method': method, 'source_selection_sha256': sha256(frozen)})
        receipt = transport.parent / 'DEV_SELECTION_RECEIPT.json'
        write_json_new(receipt, {'prediction_sha256': sha256(transport), 'Fine_access': False})
        manifest['methods'][method] = {'selection': str(frozen.relative_to(out)),
                                      'selection_sha256': sha256(frozen),
                                      'transport': str(transport.relative_to(out)),
                                      'transport_sha256': sha256(transport),
                                      'receipt_sha256': sha256(receipt),
                                      'n_videos': len(videos), 'K_total': sum(r['K'] for r in videos.values())}
    write_json_new(Path(out) / 'frozen_manifest.json', manifest)
    print('FROZEN_BEFORE_GOLD', len(methods), sha256(Path(out) / 'frozen_manifest.json'), flush=True)


def check_frozen(out):
    """逐文件SHA复核，防止冻结后结果变化；不重新选择。"""
    # main、run_native_evaluation、write_tables 都调用。out 指向已有 frozen_manifest.json。
    # 返回 manifest 字典，调用者用 methods 中的相对路径找到选点、传输和评价文件。
    # 这里核对 selection/transport/receipt 的文件 SHA，不要求当前源码 SHA 等于历史源码 SHA。
    # 因此本轮新增注释不会把旧选点 SHA 判错；旧清单里的历史源码 SHA 应保持原样。
    manifest = read_json(Path(out) / 'frozen_manifest.json')
    for row in manifest['methods'].values():
        for key in ['selection', 'transport']:
            if sha256(Path(out) / row[key]) != row[key + '_sha256']:
                raise ValueError('FROZEN_FILE_CHANGED')
        if sha256((Path(out) / row['transport']).parent / 'DEV_SELECTION_RECEIPT.json') != row['receipt_sha256']:
            raise ValueError('FROZEN_RECEIPT_CHANGED')
    return manifest


def run_native_evaluation(paths, out, tolerance_s=0.0):
    """核对冻结SHA→原evaluate.py targets→30次score；不另写命中公式。"""
    # main 调用；paths['annotations'] 指向官方标注目录，out 是已冻结本轮选点的实验目录。
    # 正常返回 None；产物是 gold/DEV23_FINE.json、各 native/<method>/evaluation.json 和三个 CSV。
    manifest = check_frozen(out)
    first = Path(out) / manifest['methods']['1017_LITE_FINAL']['transport']
    targets = Path(out) / 'gold/DEV23_FINE.json'
    # first 仅为确定评价视频范围；targets 子进程此时才打开官方 DEV Fine 并导出区间。
    # Ctrl+单击 command 看子进程机制；再打开 evaluate.py 的 main→targets_command，看 Gold 来源。
    command([sys.executable, ROOT / 'evaluate.py', 'targets', '--predictions', first,
             '--annotations', paths['annotations'], '--out', targets], out, 'native_targets')
    for method, row in manifest['methods'].items():
        pred = Path(out) / row['transport']
        command([sys.executable, ROOT / 'evaluate.py', 'score', '--predictions', pred,
                 '--targets', targets, '--out', pred.parent / 'evaluation.json',
                 '--tolerance', tolerance_s], out, 'score_' + method)
    # score 子进程：evaluate.main→score_command→multi_video_report→one_hit_rate→hit_indicators。
    # hit_indicators 按 tolerance_s 外扩真实区间 [a,b]，再检查其中是否至少有一个实际帧时间。
    # 每区间有至少一帧就记命中1；Micro=总命中/总区间，Macro=各视频命中比例的平均。
    # 此处不传 --input 检查池成员，因为实际 MP4 帧时间经 snap 后可能不等于原候选时间。
    write_tables(out)
    # Ctrl+单击 write_tables：读取刚产生的真实评价 JSON 汇总；没有复制历史 CSV 作为本轮评价。


def csvout(path, rows):
    """实际评价行→CSV；只做表格，不计算新的评价定义。"""
    # write_tables 调用；rows 是行字典列表，第一行的键决定列顺序。正常返回 None。
    # x 模式拒绝覆盖；utf-8-sig 便于 Windows 表格软件显示中文，newline='' 避免多余空行。
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_tables(out):
    """原评价JSON→30行明细、12行主表、690行逐视频配对结果；随机族报告全部种子。"""
    # run_native_evaluation 调用；metrics 是 方法名→原 score 生成的 time_interval_metrics 字典。
    # 这里的 hits/rate/micro/macro 都读取原评价输出，只做百分比、差值和随机种子汇总。
    manifest = check_frozen(out)
    metrics = {m: read_json(Path(out) / r['transport'].replace('DEV_METHOD_FROZEN.json', 'evaluation.json'))['time_interval_metrics']
               for m, r in manifest['methods'].items()}
    lite = metrics['1017_LITE_FINAL']
    rows, each = [], []
    for method, stats in metrics.items():
        if stats['total_intervals'] != 10344:
            raise ValueError('OFFICIAL_FINE_COUNT_CHANGED')
    # 10344 是完整 DEV23 的 Fine 区间总数。run --video 虽能单视频选择/评分，仍会在此检查失败；
    # 它不能产出正式完整主表。局部调试应使用 select --video，只到冻结，不调用此函数。
        selection = read_json(Path(out) / manifest['methods'][method]['selection'])
        info = selection.get('input_information', INPUT_INFORMATION.get(method,
               'TIME_4K' if method.startswith('RANDOM_4K_') else 'OFFICIAL_ALL_FRAME_TIME'))
        source_id = sha256(Path(out) / 'train/HGKS_LITE_MODEL.joblib') if method == '1017_LITE_FINAL' else sha256(ROOT / 'baselines.py')
    # SOURCE_ID 是本次输出所引用的模型/源码文件身份；新增注释会使今后输出的源码身份变化，
    # 这与选帧时间、帧号、Fine 命中是否相等是不同的检查，不能改旧结果来抹平 SHA 差异。
        authority = {'SOURCE_ID': source_id, 'input_information': info,
                     'CLIP_model_id': 'openai/clip-vit-base-patch32' if 'CLIP512' in info else '',
                     'HSV_input': 'REAL_RGB_HSV40' if 'HSV40' in info else '',
                     'Oracle_permissions': 'OBJECT+COARSE+TRAIN_FINE' if method == '1017_LITE_FINAL' else
                                           'COARSE' if method == 'INVERSE_EXP_SOP_STAGE' else 'NONE',
                     'comparison_scope': 'SYSTEM_LEVEL_SAME_DEV_SAME_K'}
        rows.append({'method_id': method, 'hits': stats['total_hits'], 'Fine_intervals': stats['total_intervals'],
                     'Micro_percent': 100*stats['micro'], 'Macro_percent': 100*stats['macro'],
                     'K_total': manifest['methods'][method]['K_total'],
                     'metric': f"Fine Action one-hit ±{stats['tolerance_seconds']:g}s", **authority})
        for v, r in stats['by_video'].items():
    # r 是一个视频的统计；delta_hits_vs_Lite 在同视频比较，正值表示比 Lite 多命中几个区间。
            each.append({'method_id': method, 'video_id': v, 'hits': r['hits'], 'Fine_intervals': r['n_intervals'],
                         'rate_percent': 100*r['rate'], 'delta_hits_vs_Lite': r['hits']-lite['by_video'][v]['hits'],
                         'K': selection['videos'][v]['K'], 'selection_sha256': manifest['methods'][method]['selection_sha256'],
                         'Micro_percent': 100*stats['micro'], 'Macro_percent': 100*stats['macro'], **authority})
    csvout(Path(out) / 'all_seed_results.csv', rows)
    csvout(Path(out) / 'per_video.csv', each)
    summary = [{**r, 'Micro_std_percent': 0.0, 'Macro_std_percent': 0.0, 'n_seeds': 1}
               for r in rows if not r['method_id'].startswith('RANDOM_')]
    for family in ['RANDOM_4K', 'RANDOM_ALL_FRAMES']:
    # 非随机10行直接保留；每个随机族将10种子平均成1行，所以主表12行而不是30行。
    # np.std(ddof=0) 是这10个已运行种子的总体标准差；均值可能为小数，不是一次运行的新命中数。
        family_rows = [r for r in rows if r['method_id'].startswith(family + '_SEED_')]
        row = dict(family_rows[0])
        row.update(method_id=family + '_10_SEEDS', hits=float(np.mean([r['hits'] for r in family_rows])),
                   Micro_percent=float(np.mean([r['Micro_percent'] for r in family_rows])),
                   Macro_percent=float(np.mean([r['Macro_percent'] for r in family_rows])),
                   Micro_std_percent=float(np.std([r['Micro_percent'] for r in family_rows], ddof=0)),
                   Macro_std_percent=float(np.std([r['Macro_percent'] for r in family_rows], ddof=0)), n_seeds=10)
        summary.append(row)
    csvout(Path(out) / 'summary.csv', summary)


def main():
    """从这里按顺序阅读；Ctrl+单击调用函数，Alt+左箭头返回继续。"""
    # 真实 PowerShell 入口（检查时输出目录尚不存在，再次执行前请更换新名字）：
    # Set-Location C:\Project08\1019
    # .\.venv\Scripts\python.exe -X utf8 run1019.py run `
    #   --out results\MY_CACHED_ANNOTATED_001 `
    #   --mode cached `
    #   --reference-selections _reference\EXPECTED_1019_SELECTIONS.json `
    #   --reference-results _reference\EXPECTED_1019_RESULTS.csv
    # run：要求 TRAIN70/DEV23/缓存/标注/参考等真实输入；新建 out，训练→预测→选择→冻结→评价。
    # select：要求 out/train 和 out/lite_raw 已就绪；只做选择和冻结，正常 return 后不评价。
    # evaluate：要求 out 已有冻结选择/清单/模型，且尚无同名 Gold/评价/表格；只核验并评价。
    # vision.py、baselines.py 是被调用模块；单独 python vision.py 不会触发整套实验。
    # --video 只供局部调试；run --video 仍训练/预测完整输入，表格阶段要求10344区间而失败。
    # Raw 是真 MP4→JPEG→HSV/CLIP，但历史 Raw 严格等价685/690，仍有5项 MaxInfo 不一致；
    # 不可修改特征、舍入、公式或 argmax 强行消除，也不能用 cached 的 PASS 代表 Raw PASS。
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['run', 'select', 'evaluate'])
    p.add_argument('--paths', type=Path, default=ROOT / 'paths.json')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--mode', choices=['cached', 'raw'], default='cached')
    p.add_argument('--tolerance', type=float, choices=(0.0, 0.5), default=0.0,
                   help='评价容差（秒）：默认0，可选0.5')
    p.add_argument('--video', help='仅调试一个DEV视频；不是正式DEV23结果')
    p.add_argument('--reference-selections', type=Path, help='仅冻结后等价检查使用')
    p.add_argument('--reference-results', type=Path, help='仅评价后等价检查使用')
    a = p.parse_args()
    # parse_args 把终端字符串解析为 a.stage/a.paths/a.out/...；随后业务首次调用自定义 load_paths。
    paths = load_paths(a.paths)
    # Ctrl+单击进入 load_paths，Alt+左箭头回来：paths 是绝对路径字典；out 是本轮绝对输出 Path。
    out = a.out.resolve()
    if a.stage == 'select':
        choose_frames(paths, out, a.mode, a.video)
    # Ctrl+单击 choose_frames。此 return 意味着 select 不执行下方的参考核验、targets、score或主表。
        return
    if a.stage == 'run':
        out.mkdir(parents=True, exist_ok=False)
    # exist_ok=False 保证新目录。下一行传给 command 的 train.py 是跨进程入口，不是函数转发。
    # Ctrl+单击 command 看到 subprocess.run 后，打开 train.py 的 main→train：
    # TRAIN70→prepare.frames 拼141维X→oracle_labels 预算Oracle产生y→sample_weights产生w→
    # HistGradientBoostingClassifier.fit(X,y,sample_weight=w)→out/train/HGKS_LITE_MODEL.joblib。
    # 16/64/141维、预算、样本权重和 HGB 均保持原五文件计算；没有在本文件复制一遍训练。
        command([sys.executable, ROOT / 'train.py', '--train', paths['train'], '--out', out / 'train'], out, 'train_fit')
        command([sys.executable, ROOT / 'predict.py', '--input', paths['dev'], '--model', out / 'train/HGKS_LITE_MODEL.joblib',
                 '--out', out / 'lite_raw', '--expected-split', 'DEV', '--expected-videos', '23',
                 '--trace-video', 'P02-20240209-231412', '--trace-time', '8.75'], out, 'lite_predict')
    # train 子进程退出才到 predict 子进程；打开 predict.main→select_video：DEV23→frames141→
    # HGB 概率→视频内排名→按原时间新颖度逐轮选K点→out/lite_raw/DEV_LITE_FROZEN.json。
    # trace-video/time 仅输出已有 P02/8.75秒解释案例，不改变 DEV23 的推理范围和选帧规则。
        args = [paths['feature_python'], ROOT / 'run1019.py', 'select', '--paths', a.paths, '--out', out, '--mode', a.mode]
    # feature_python 使用历史 Baseline 的 NumPy2.4.0/sklearn1.9.1；当前 .venv 用于原 Lite 环境。
    # 同一 run1019.py 再启动 select，只为依赖版本；子进程 main→choose_frames 直接调用真实算法。
        if a.video:
            args += ['--video', a.video]
        command(args, out, 'all_baselines_select')
    # select 子进程完成后回到父进程；evaluate stage 则直接从这里开始，只读取已有冻结文件。
    check_frozen(out)
    if a.reference_selections:
    # 这时30套选择已经冻结，才读取历史参考核验请求时间/实际时间/帧号；失败即停止评价。
        command([sys.executable, ROOT / 'tests/check_equivalence.py', '--run', out,
                 '--expected-selections', a.reference_selections], out, 'selection_equivalence')
    run_native_evaluation(paths, out, a.tolerance)
    # Ctrl+单击 run_native_evaluation，跟到原 evaluate.py；返回时本轮评价 JSON、三个 CSV 已写出。
    if a.reference_results:
    # 只有重新 score 之后才用历史分数核验；这不会把旧CSV复制成新的分数。
        command([sys.executable, ROOT / 'tests/check_equivalence.py', '--run', out,
                 '--expected-results', a.reference_results], out, 'evaluation_equivalence')
    print('RUN_COMPLETE', out, flush=True)


if __name__ == '__main__':
    # 直接运行本文件时 __name__ 为 '__main__'；被 import 时不自动训练/选帧/评价。
    main()
