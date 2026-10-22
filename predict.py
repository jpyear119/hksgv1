"""1018 Lite inference: ONE fitted HGB rank + time diversity -> fixed K times.

No event discovery, HMM, SOP matching, 16384D Logistic or coverage matrix.
All 16D/64D/141D inputs are still prepared by the original 1017 prepare.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np

from data import load_inference, read_json, sha256, write_json_new, digest
from prepare import budget, frames

MODEL_VERSION='HGKS_LITE_SINGLE_HGB_V1'
FEATURE_VERSION='HGKS_FRAME_141_FINAL_V1'

# Fixed *before* selection: exact historical ratio of the time and HGB terms.
# 1.25*0.724128... is the prior TRAIN-heldout reliability-weighted HGB coefficient.
HGB_WEIGHT=1.25*0.7241283124595239
TIME_WEIGHT=0.7


def rank_in_this_video(probabilities):
    """Convert arbitrary HGB probabilities into ranks from 0 (low) to 1 (high)."""
    p=np.asarray(probabilities,dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError('NONFINITE_HGB_SCORES')
    if max(p)-min(p)<1e-8:
        return np.full(len(p),0.5)
    order=np.argsort(p,kind='stable')
    rank=np.empty(len(p))
    rank[order]=np.arange(len(p))/max(len(p)-1,1)
    return rank


def select_video(pool, model, duration_s, trace=None):  #这里是筛选的代码
    """Straight-line version of the complete K-round selection algorithm."""
    if model['bundle_version']!=MODEL_VERSION or model['feature_version']!=FEATURE_VERSION:
        raise ValueError('WRONG_LITE_MODEL_VERSION')
    times=np.array([float(row['t']) for row in pool])
    if not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0):
        raise ValueError('BAD_CANDIDATE_TIMES')
    K=budget(duration_s)
    if len(pool)<K:raise ValueError('FEWER_CANDIDATES_THAN_BUDGET')
    # 2. 141维输入 HGB，得到候选的历史排名；它不是动作真值概率。
    X=frames(pool)       # Original 1017 X141, not a newly invented feature vector.
    HGB_probability=model['anchor'].predict_proba(X)[:,1]
    HGB_rank=rank_in_this_video(HGB_probability)

    available=np.ones(len(pool),dtype=bool)
    nearest_time_distance=np.full(len(pool),np.inf)
    chosen=[]
    chosen_scores={}
    trace_rows=[]
    time_scale=max(1.0,float(times[-1])/K)

    for step in range(K):
        # 1 means candidate is far enough from all previously chosen timestamps.
        time_novelty=np.minimum(1.0,nearest_time_distance/time_scale)
        time_novelty[~np.isfinite(time_novelty)]=1.0
        score=HGB_WEIGHT*HGB_rank+TIME_WEIGHT*time_novelty
        score[~available]=-np.inf
        # 4. 选中当前最大分数的候选，更新覆盖量和已选时间距离。
        i=int(np.argmax(score))       # The best still-unused candidate this round.
        chosen.append(i)
        chosen_scores[i]={'history_rank':float(HGB_rank[i]),
                          'time_novelty':float(time_novelty[i]),
                          'final_score':float(score[i])}
        available[i]=False
        nearest_time_distance=np.minimum(nearest_time_distance,np.abs(times-times[i]))
        trace_rows.append({'round':step+1,'winner_index':i,'winner_time_s':float(times[i]),
                           'HGB_rank':float(HGB_rank[i]),'time_novelty':float(time_novelty[i]),
                           'final_score':float(score[i])})

    result=[{'time_s':float(times[i]),'candidate_index':int(i),
             'source':pool[i]['source'],'object_names':pool[i].get('object_names',[]),
             'score_terms':chosen_scores[i],
             'selection_reason':'HGB_RANK_PLUS_TEMPORAL_DIVERSITY'} for i in sorted(chosen)]
    if trace is not None:
        q=trace.get('candidate_index', chosen[0])
        trace.update(video_id=pool[q]['video_id'],time_s=float(times[q]),
                     candidate_index=int(q),F0_F15=pool[q]['features'],
                     HGB_probability=float(HGB_probability[q]),HGB_rank=float(HGB_rank[q]),
                     selection_order=trace_rows)
    return result,{'K':K,'n_input_candidates':len(pool),
                   'model_components':['HGB'],'algorithm_version':MODEL_VERSION,
                   'no_DEV_TEST_FINE_or_QA_used':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--expected-split',choices=['DEV','TEST'])
    parser.add_argument('--expected-videos',type=int)
    parser.add_argument('--trace-video')
    parser.add_argument('--trace-time',type=float)
    args=parser.parse_args()
    if args.out.exists():raise SystemExit('OUTPUT_EXISTS: '+str(args.out))
    pools,durations,_stages,_sop,_provenance,split=load_inference(args.input,args.expected_split)
    if args.expected_videos is not None and len(pools)!=args.expected_videos:
        raise ValueError('VIDEO_COUNT_MISMATCH')
    receipt_path=args.model.parent/'TRAIN_RECEIPT.json'
    receipt=read_json(receipt_path)
    if sha256(args.model)!=receipt['model_sha256']:
        raise ValueError('MODEL_SHA256_MISMATCH')
    # joblib is executable deserialization. Only load your own trusted artifact.
    model=joblib.load(args.model)
    videos=sorted(pools)
    focus=args.trace_video or videos[0]
    if focus not in pools:raise ValueError('TRACE_VIDEO_NOT_FOUND')
    trace={}
    pred={}
    for video in videos:
        local_trace=trace if video==focus else None
        if local_trace is not None and args.trace_time is not None:
            local_trace['candidate_index']=min(range(len(pools[video])),
                key=lambda i:abs(pools[video][i]['t']-args.trace_time))
        rows,diagnostics=select_video(pools[video],model,durations[video],trace=local_trace)
        pred[video]={'K':len(rows),'duration_s':durations[video],
                     'selected':rows,'diagnostics':diagnostics,
                     'candidate_times_digest':digest([r['t'] for r in pools[video]])}
    args.out.mkdir(parents=True,exist_ok=False)
    output=args.out/(split+'_LITE_FROZEN.json')
    write_json_new(output,{'schema':'HGKS_FINAL_PREDICTION_V1','algorithm_version':MODEL_VERSION,
                          'split':split,'model_sha256':sha256(args.model),
                          'input_sha256':sha256(args.input),'method':'HGB_TEMPORAL',
                          'n_videos':len(pred),'predictions':pred,'QA_or_Fine_at_inference':False})
    trace.update(input_sha256=sha256(args.input),model_sha256=sha256(args.model),
                 prediction_sha256=sha256(output))
    write_json_new(args.out/'CANDIDATE_TRACE.json',trace)
    write_json_new(args.out/(split+'_SELECTION_RECEIPT.json'),{
        'status':'FROZEN_UNLABELED_INFERENCE','algorithm_version':MODEL_VERSION,
        'model_sha256':sha256(args.model),'input_sha256':sha256(args.input),
        'prediction_sha256':sha256(output),'training_status':receipt['status']})
    print('LITE_SELECTED',len(pred),'videos',sum(r['K'] for r in pred.values()),'timestamps',output)

if __name__=='__main__':
    main()
