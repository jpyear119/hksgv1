"""1018 Lite training: TRAIN70 candidates + TRAIN Fine -> ONE HGB model.

Only the data-defined 141D features and the original budget-Oracle target are used.
No pairwise samples, 16384-dimensional hash features, Logistic heads or HMM.
"""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from data import load_train, sha256, write_json_new
from prepare import budget, frames

MODEL_VERSION = 'HGKS_LITE_SINGLE_HGB_V1'
FEATURE_VERSION = 'HGKS_FRAME_141_FINAL_V1'


def oracle_labels(candidates, actions, budget_k, tolerance_s=0.55):  #定义label，pool是候选点列表。fine是细粒度的标注，包含事件，margin运行判断覆盖的时候向外扩展多少事件，这里默认0.55
    """1 marks TRAIN candidate chosen by the original K-budget Fine Oracle."""
    times = np.array([c['t'] for c in candidates], dtype=float)  #这里提取所有的候选事件
    #这里是覆盖度，允许有冗余空间
    visible_actions = [set() for _ in candidates]  #这里定义label里面的细粒度的动作覆盖度
    valid_actions = sorted((float(a['a']), float(a['b'])) for a in actions if float(a['b']) > float(a['a']))
    if not valid_actions:
        raise ValueError('TRAIN_FINE_EMPTY')
    for j, (start, end) in enumerate(valid_actions):
        left = np.searchsorted(times, start - tolerance_s)
        right = np.searchsorted(times, end + tolerance_s, side='right')
        for i in range(left, right):
            visible_actions[i].add(j)

    labels = np.zeros(len(candidates), dtype=np.int8)
    already_covered = set()  #这里表示用过的编号
    chosen = []
    quality = np.array([c.get('cue_quality', 0.0) for c in candidates], dtype=float)
    for _ in range(min(budget_k, len(candidates))):
        best, best_score = -1, -float('inf')
        for i, action_ids in enumerate(visible_actions):
            if labels[i]:
                continue
            new_actions = len(action_ids - already_covered)  #这里计算覆盖的新增动作
            if not new_actions:
                continue
            min_distance = min((abs(times[i] - times[j]) for j in chosen), default=1e6)  #这里是计算候选离以选点的距离
            score = 100*new_actions + 0.01*min(min_distance, 30) + 0.001*quality[i]
            if score > best_score:
                best, best_score = i, score
        if best < 0:
            break
        labels[best] = 1
        chosen.append(best)
        already_covered.update(visible_actions[best])
    return labels


def sample_weights(labels, video_ids):
    """The same video and class balance rule as the original TRAIN70."""
    ct = collections.Counter(video_ids)  #计算视频的候选
    weights = np.array([1/max(ct[vid], 1) for vid in video_ids], dtype=float)
    weights /= max(np.mean(weights), 1e-12)
    positives = max(int(np.sum(labels)), 1)  #p是正面例子的数量，就是label里面是1的
    negatives = max(len(labels)-positives, 1)  #n是负面例子，label里面是0的
    ratio = np.sqrt(negatives/positives)
    weights[labels==1] *= min(12, max(0.5, ratio))  #这里是个bool语法，强调正例子和负例子的权重。因为正面例子太少了，所以要赋予更高的权重以得到重视
    weights[labels==0] *= min(12, max(0.5, 1/max(ratio, 1e-10)))
    return weights


def train(pools, fines, groups, durations, strict=True):  #具体的训练代码
    videos = sorted(pools)
    if set(videos) != set(groups) or set(videos) != set(durations):
        raise ValueError('TRAIN_VIDEOS_GROUPS_DURATIONS_DIFFER')
    if strict and (len(videos)!=70 or len(set(groups.values()))!=9):
        raise ValueError('TRAIN70_9_GROUPS_REQUIRED')
    if not set(fines).issubset(pools):
        raise ValueError('FINE_OUTSIDE_TRAIN')
    # 1. 每个训练视频：141维 X 与预算 Oracle 标签 y（1=该候选被 Oracle 选择）。
    feature_parts = []  #保存各视频的141维矩阵
    label_parts = []  #保存各个视频的0/1的label
    video_ids = []
    for vid in videos:
        if len(pools[vid]) < 5 or not fines.get(vid):
            continue
        #这里是对齐输入
        features = frames(pools[vid])         # X: n_candidates × 141
        #这里定义了label
        labels = oracle_labels(pools[vid], fines[vid], budget(durations[vid]))  # y: 0 or 1
        feature_parts.append(features)
        label_parts.append(labels)  #添加这些候选的label标签
        video_ids.extend([vid]*len(labels))  #创造id以保证对齐
    if len(feature_parts)<5:
        raise ValueError('INSUFFICIENT_LABELED_TRAIN_VIDEOS')
    X = np.vstack(feature_parts)  #pool的矩阵，吧每个视频的候选特征链接。np.vstack() 沿着行方向拼接二维矩阵
    y = np.concatenate(label_parts)  #把一维标签数组接起来。
    w = sample_weights(y, video_ids)  #这里是帮助视频长度不同的视频更换权重而防止提供过多的例子给长视频
    if X.shape[1]!=141 or len(np.unique(y))!=2:
        raise ValueError('TRAIN_FEATURE_OR_LABEL_SHAPE')

    # Original HGB configuration and fit; no other models are fitted.
    # 3. 141维 X、0/1 y、样本权重 → 全 TRAIN HGB.fit。
    model = HistGradientBoostingClassifier(  #这里是全量跑的
        max_leaf_nodes=23, max_depth=6, learning_rate=0.065,
        l2_regularization=2.0, max_iter=135,
        min_samples_leaf=max(12, min(55, int(len(y)*0.0007))),
        early_stopping=False, random_state=20261011, loss='log_loss',
    )
    print('TRAIN X / y / weights',X.shape,y.shape,w.shape,flush=True)
    model.fit(X, y, sample_weight=w)
    return {'bundle_version':MODEL_VERSION,'feature_version':FEATURE_VERSION,
            'anchor':model,'training_scope':'TRAIN_FINE_ONLY'}, {
        'training_videos':len(feature_parts),'X_shape':list(X.shape),
        'positive_oracle_candidates':int(y.sum()),
        'negative_candidates':int((y==0).sum()),'HGB_parameters':model.get_params(),
        'n_estimators':int(model.n_iter_),
        'target':'TRAIN_FINE_BUDGET_ORACLE', 'no_DEV_TEST_Fine_used':True,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--allow-small-fixture',action='store_true')
    args=parser.parse_args()
    if args.out.exists():raise SystemExit('OUTPUT_EXISTS: '+str(args.out))
    pools,fines,groups,durations,provenance=load_train(args.train,strict70=not args.allow_small_fixture)
    bundle,report=train(pools,fines,groups,durations,strict=not args.allow_small_fixture)  #这里开始训练
    args.out.mkdir(parents=True,exist_ok=False)
    target=args.out/'HGKS_LITE_MODEL.joblib'
    joblib.dump(bundle,target,compress=3)
    status=('SYNTHETIC_ENGINEERING_FIT' if args.allow_small_fixture
            else 'FITTED_FROM_PORTABLE_TRAIN_INPUT')
    write_json_new(args.out/'TRAIN_RECEIPT.json', {
        'status':status,'scope':'TRAIN_ONLY','artifact':target.name,
        'model_sha256':sha256(target),'train_sha256':sha256(args.train),
        'input_provenance':provenance,'model_report':report,
        'sklearn_version':sklearn.__version__,
        'execution_utc':datetime.now(timezone.utc).isoformat(),
        'training_input_no_DEV_TEST_Fine':True,
    })
    print('MODEL_SAVED',target,flush=True)

if __name__=='__main__':
    main()
