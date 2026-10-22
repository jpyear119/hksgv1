"""Portable, deterministic reproductions of the frozen 2026-10-13 external baselines.

IMPORTANT. Names ending *_ADAPT are not official reimplementations of entire
published pipelines. The original private result ZIP, not a rerun with changed
dependencies, remains the authoritative record for the 2026-10-13 scores.

No Fine/QA reference annotation is imported anywhere in this module.
"""
from __future__ import annotations
# 这份文件是被调用的选帧模块，没有命令行main入口。整套实验从run1019.py的main()开始。
# choose_frames()把真实特征交给all_indices()，这里返回下标；返回后才用times[下标]取秒数。
# 五种视觉规则是：HSV的VSUMM/MaxVol、CLIP的MaxInfo/MaxInfo-Chunk、SOP查询的AKS。
# 这里的“适配”表示当前HD-EPIC版本，不能说是各篇论文全部流程的原样实现。
import math
import numpy as np

VERSION='EXTERNAL_BASELINE_FIXED4K_ADAPTED_20261013'
# SEED是历史固定随机种子，不是训练学到的数。完整运行还有两族各10个预设种子。
# 10个种子是同一个Random规则的10次固定重复，不是10种算法；有效种子还加视频ID的字符和。
SEED=20261012
METHODS=(
  'UNIFORM_FULL_VIDEO','RANDOM_K','INVERSE_EXP_SOP_STAGE',
  'VSUMM_HIST_KMEANS','MAXVOL_HIST_DOPT',
  'MAXINFO_CLIP_MAXVOL_ADAPT','MAXINFO_CHUNK_CLIP_ADAPT','AKS_SOP_CLIP_ADAPT',
)


def grid(duration_s:float,K:int,ratio:int=4)->np.ndarray:
    """Common pre-registered full-video candidate time grid of ratio*K points."""
    # 谁调用：run1019.choose_frames()；输入是DEV便携文件的时长（秒）和prepare.budget得到的K。
    # ratio=4是历史人工固定的预采样倍数，优化来源未核实，不是HGB学到的参数。
    # np.arange(ratio*K)产生0到ratio*K-1；加0.5取每个等长时间格的中心，避开视频两端。
    # 返回一维float数组，赋给choose_frames的times。P08的K=40，所以times有160个秒数。
    # P08真实时长1184.51197917秒，times[2]=18.50799967453125秒；下标2尚不是MP4帧号。
    if duration_s<=0 or K<1 or ratio<2: raise ValueError('BAD_VIDEO_OR_GRID_BUDGET')
    return (np.arange(ratio*K,dtype=float)+.5)*duration_s/(ratio*K)


def _check_indices(idx,n,K,name):
    # 各选择函数在返回前调用这里：idx为候选下标，n为候选总数，K为应选数量。
    # 转Python整数后同时检查数量、去重和范围；任一条件不满足就报错，不默默补点。
    # sorted只按时间网格下标排序，便于调用方映射时间；返回顺序不代表贪心选择的先后。
    arr=[int(i) for i in idx]
    if len(arr)!=K or len(set(arr))!=K or any(i<0 or i>=n for i in arr):
        raise ValueError(name+'_NONUNIQUE_OR_WRONG_K')
    return sorted(arr)


def uniform_indices(n,K):
    """Closest distinct candidate for each bin midpoint."""
    # 输入只有候选数量n和预算K，不读取RGB或Gold；all_indices和choose_frames都调用它。
    # (i+.5)*n/K-.5把第i个均匀格的中心换算成零起点候选下标，再用round取最近整数。
    # Python的round在恰好一半时取偶数，这也是现有选点规则的一部分，不能换成floor。
    # 返回K个下标：在4K网格中就索引times，在全帧控制中就索引官方clock；两个空间不同。
    return _check_indices([min(n-1,int(round((i+.5)*n/K-.5))) for i in range(K)],n,K,'UNIFORM')


def random_indices(n,K,seed):
    # 只按时间候选下标随机抽取，不看图像；replace=False表示不放回，保证不重复。
    # seed固定后，在相同输入和依赖环境中重建相同随机数序列，不是每次换一个“更好”的种子。
    # 返回K个排序下标给all_indices或choose_frames，后者取秒数、snap、冻结后统一评价。
    return _check_indices(np.random.default_rng(seed).choice(n,K,replace=False),n,K,'RANDOM')


def _stage_labels(times,coarse):
    # inverse_exp_indices调用：times是一维候选秒数，coarse来自DEV输入的官方粗阶段列表（Oracle）。
    # valid保存(a,b,j)：阶段起止秒数及其在coarse中的编号；无合法阶段时全部归到编号0。
    valid=[]
    for j,stage in enumerate(coarse):
        try: a,b=float(stage['start_s']),float(stage['end_s'])
        except (KeyError,ValueError,TypeError): continue
        if b>a: valid.append((a,b,j))
    if not valid: return np.zeros(len(times),dtype=int)
    output=[]
    for t in times:
        # 同时命中多个阶段时，min((时长,编号))先取最短阶段，再按编号打破相同情况。
        hit=[(b-a,j) for a,b,j in valid if a<=t<b]
        if hit: output.append(min(hit)[1]);continue
        # 没有命中任何阶段时，归到起点或终点离t最近的阶段；这是现有适配规则。
        output.append(min(valid,key=lambda q:min(abs(t-q[0]),abs(t-q[1])))[2])
    # 返回与times等长的整数数组，每个位置表示这个候选分到哪个阶段。
    return np.asarray(output,dtype=int)


def _allocate_quotas(groups,K,total):
    # 谁调用：InverseExp和MaxInfo-Chunk。groups是{组编号:该组的候选下标数组}。
    # total是所有组的候选总量；返回{组编号:整数预算}，预算总和必须恰好为K。
    # keys固定处理顺序；sizes记录每组有多少候选，不能给一个组超过它容量的预算。
    keys=sorted(groups); sizes=np.asarray([len(groups[k]) for k in keys],int)
    if total!=sum(sizes) or K>total: raise ValueError('BAD_GROUP_ALLOCATION')
    desired=K*sizes/total
    # desired是按候选数量比例分配的浮点理想预算；floor先向下取整，再处理余数。
    allocated=np.minimum(sizes,np.floor(desired).astype(int))
    # 若总K足够，先让每个非空组至少得到1个；这一步可能使预算暂时超出K。
    if K>=len(keys):allocated=np.maximum(allocated,np.minimum(sizes,1))
    while allocated.sum()>K:
        # max的key是元组：先找“超过理想值最多”的组，相同则看已分配量，减去1。
        i=max((i for i in range(len(keys)) if allocated[i]>0),key=lambda i:(allocated[i]-desired[i],allocated[i]))
        allocated[i]-=1
    while allocated.sum()<K:
        # 还有预算时先补“距离理想值最多”的组，再看剩余容量；-i使并列时先处理前面的组。
        i=max((i for i in range(len(keys)) if allocated[i]<sizes[i]),key=lambda i:(desired[i]-allocated[i],sizes[i]-allocated[i],-i))
        allocated[i]+=1
    # zip把组编号和对应预算配对成字典；返回后由调用方在各组内部真正选点。
    return dict(zip(keys,allocated.tolist()))


def inverse_exp_indices(times,coarse,K,seed,lam=3.0):
    """ICPR'26 inverse-exponential mechanism, adapted within coarse SOP stages.

    Allocate K approximately proportional to valid stage candidate counts. For
    each stage, fix its last point and sample other q-1 without replacement
    with exponentially increasing probability toward stage end. **Adapted**,
    not the author's original fixed-10-second anticipation protocol.
    """
    # all_indices调用：只用4K时间网格和coarse阶段，不读取HSV、CLIP或Fine。
    # 思想：先按各阶段候选数分K，再在每阶段偏向末端抽样；不是论文完整的固定10秒协议。
    # lam=3.0为当前源码继承的指数强度，人工固定，优化来源未核实；数值越大越偏向阶段末尾。
    labels=_stage_labels(times,coarse)
    # np.unique取阶段编号；flatnonzero得到属于该阶段的位置，groups的值仍是全网格下标。
    groups={int(k):np.flatnonzero(labels==k) for k in np.unique(labels)}
    quotas=_allocate_quotas(groups,K,len(times))
    rng=np.random.default_rng(seed);chosen=[]
    for key,ids in groups.items():
        q=quotas[key]
        if not q: continue
        if q>=len(ids):chosen.extend(ids.tolist());continue
        if q==1:chosen.append(int(ids[-1]));continue
        # 每阶段必须固定最后一个候选；other排除该点，其余q-1个才随机抽取。
        other=ids[:-1];pos=np.linspace(0,1,len(ids))[:-1]
        # pos从阶段前端0走向末端1；exp(lam*(pos-1))随pos增大而增大，所以偏向末端。
        # 减logits.max()让指数值更稳定，不改变除以总和之后的概率比例。
        logits=lam*(pos-1.);weights=np.exp(logits-logits.max());weights/=weights.sum()
        chosen.extend(rng.choice(other,size=q-1,replace=False,p=weights).tolist())
        chosen.append(int(ids[-1]))
    # 每组恰好q个且组间不重叠；辅助函数最后检查全视频恰好K个，返回给all_indices。
    return _check_indices(chosen,len(times),K,'INVERSE_EXP')






def vsumm_indices(features,K,seed=SEED):
    """VSUMM-style nearest unique KMeans representatives, not original full VSUMM."""
    # all_indices传入features=hist，即JPEG计算的HSV40。P08是160行×40列，不用SOP/Fine。
    # 基本思想：把颜色特征聚成K组，再给每个聚类中心找一张最近、尚未使用的代表帧。
    from sklearn.cluster import KMeans
    x=np.asarray(features,np.float32);n=len(x)
    if K>=n:return list(range(n))
    # KMeans的中心是40维均值向量，不一定对应某张真实图片；中心由当前视频特征拟合。
    # n_init=4是重复初始化次数，max_iter=120是每次迭代上限，seed是固定随机种子。
    # 这些是现有本地VSUMM适配的固定设置，优化来源未核实，不是全套原版VSUMM流程。
    km=KMeans(n_clusters=K,random_state=seed,n_init=4,max_iter=120).fit(x)
    # 广播：x[:,None,:]为(n,1,40)，中心[None,:,:]为(1,K,40)，相减得到(n,K,40)。
    # 平方后sum(axis=2)把40维加起来，dist[i,cluster]就是候选i到该中心的平方距离。
    dist=((x[:,None,:]-km.cluster_centers_[None,:,:])**2).sum(axis=2)
    used=set();chosen=[]
    for cluster in range(K):
        # argsort按距离由小到大返回候选下标；stable保留同距离候选的原顺序。
        # next找第一个不在used中的点，避免两个中心选到同一帧；每个中心贡献1个，共K个。
        i=next(int(i) for i in np.argsort(dist[:,cluster],kind='stable') if i not in used)
        chosen.append(i);used.add(i)
    return _check_indices(chosen,n,K,'VSUMM')


def svd_reduced(x,K,rank=32):
    """Row-normalized U*S^0.3, not native official MaxInfo CLIP-L/14-336 SVD setup."""
    # maxvol_indices调用：输入x是n×d矩阵，行是图像候选；d为HSV的40或CLIP的512。
    # 先转float64，再按列减均值，让分解描述视频内部差异；这里不使用标签。
    x=np.asarray(x,dtype=np.float64)
    x=x-x.mean(axis=0,keepdims=True)
    if not np.isfinite(x).all():raise ValueError('NONFINITE_FEATURES')
    # SVD（奇异值分解）把x写成U@diag(s)@Vt；full_matrices=False只保留需要的部分。
    # p=min(n,d)时，u为(n,p)、s为(p,)、第三项为(p,d)；这里用_接住不用的Vt。
    u,s,_=np.linalg.svd(x,full_matrices=False)
    r=max(1,min(rank,K,len(x)-1,x.shape[1]))
    # r受32、K、n-1、d共同限制。P08全视频HSV/CLIP的r都为32，返回z为160×32。
    # s[:r]是一维r个数，广播到u每一行；s**0.3调整各方向尺度，不是学习得到的指数。
    # rank=32、指数0.3及这些数值下界是当前源码固定参数，优化来源未核实。
    z=u[:,:r]*np.maximum(s[:r],1e-7)**.3
    # norm(axis=1)求每行欧氏长度；keepdims=True保留(n,1)，便于每行分别相除。
    # 得到单位长度行向量后，后面的贪心评分主要比较方向；1e-9避免除零。
    z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-9)
    return z


def _d_opt_logdet_indices(z,K,prior=None):
    """K-round regularized greedy logdet; **not** official rect_maxvol pivots."""
    # maxvol_indices调用；z是SVD后的n×d矩阵，返回K个候选下标，不输出时间。
    # 这是本地D-opt/Logdet适配：逐轮增加描述不同特征方向的点，不是MaxInfo官方完整算法。
    # 可把A理解为0.6*单位矩阵+已选行向量的外积之和；inv保存A的逆矩阵。
    # 0.6为固定正则强度，使初始A可逆；优化来源未核实，不能按本轮DEV分数调整。
    X=np.asarray(z,float);n,d=X.shape
    if K>=n:return list(range(n))
    inv=np.eye(d)/.6;valid=np.ones(n,bool);chosen=[]
    # prior为可选的每候选加分；现有maxvol_indices调用没有传它，所以这里全为0。
    if prior is None:prior=np.zeros(n)
    for _ in range(K):
        # X@inv是矩阵乘法，结果a仍为(n,d)；第i行对应x_i乘当前逆矩阵。
        a=X@inv
        # einsum('ij,ij->i')逐行点乘再求和，得到n个x_i^T A^-1 x_i分数。
        # 加入该点的logdet增量为log(1+这个量)，单调性使此处可直接比较这个量。
        score=np.einsum('ij,ij->i',a,X)+prior
        score[~valid]=-np.inf
        # ~valid是“已经选过”的布尔位置；设负无穷后不可能再赢。argmax取最大值下标。
        # chosen记录贪心先后，最后_check_indices才排序；不要把排序输出当作第一轮赢家。
        i=int(np.argmax(score));chosen.append(i);valid[i]=False
        # v=inv@X[i]为d维向量；outer(v,v)为d×d外积。原秩1公式更新逆矩阵，避免每轮重求逆。
        # 分母1+x_i^T inv x_i对应新加入的外积；1e-10为已有防除零下界。
        v=inv@X[i];inv-=np.outer(v,v)/max(1.+float(X[i]@v),1e-10)
    # 真实限制：单位行向量使首轮理论评分相同，约1e-15的浮点差也可能影响argmax。
    # Raw中约3～5e-7的CLIP差异已导致5/690项不一致；不能加舍入或改并列规则来追一致。
    return _check_indices(chosen,n,K,'LOGDET_MAXVOL_APPROX')


def maxvol_indices(features,K):
    # all_indices可传HSV或CLIP，Chunk也复用这里；没有第二份SVD/Logdet实现。
    # Ctrl+单击svd_reduced看n×原维数怎样变成n×r，Alt+←回来再看_d_opt_logdet_indices。
    # 两步返回K个整数下标，交还调用者；方法名称取决于调用方传的是HSV还是CLIP。
    return _d_opt_logdet_indices(svd_reduced(features,K),K)


def maxinfo_chunk_indices(features,K,nchunks=16):
    # all_indices传入按时间排列的CLIP矩阵；先分时间块，让预算分散到视频各部分。
    # nchunks=16是当前适配固定值，优化来源未核实；最多分min(16,K)块，避免块数大于预算。
    x=np.asarray(features)
    bins=np.array_split(np.arange(len(x)),min(nchunks,K))
    # array_split按连续下标切分，块大小允许相差1；groups保存“块编号→全视频下标”。
    groups={i:np.asarray(b) for i,b in enumerate(bins)}
    alloc=_allocate_quotas(groups,K,len(x));selected=[]
    for i,ids in groups.items():
        quota=alloc[i]
        if quota==len(ids):selected.extend(ids.tolist())
        elif quota:
            # x[ids]取出本块矩阵；sub是块内下标，必须用ids[j]还原成全视频下标。
            sub=maxvol_indices(x[ids],quota)
            selected.extend(int(ids[j]) for j in sub)
    # 各块不相交且预算和为K；合并后检查、排序返回all_indices，再统一取时间并评价。
    return _check_indices(selected,len(x),K,'MAXINFO_CHUNK')


def aks_segments(scores,K,t1=.8,t2=-100.,max_depth=5):
    """Official AKS-style meanstd split criterion, with fixed K adapted fill."""
    # aks_indices传入每帧与SOP查询的相似度分数（一维n个数），返回(片段下标,深度)列表。
    # rec是本函数内的递归：同一个函数处理当前片段，再按时间顺序分前半/后半继续。
    # .8、-100和最大深度5沿用现有实现，优化来源未核实；不据此宣称原论文完整复刻。
    out=[]
    def rec(ids,depth):
        # ids为本段在全视频网格的位置；vals为本段分数，n限制最多比较K个高分候选。
        vals=scores[ids];n=min(K,len(vals))
        top=np.partition(vals,-n)[-n:]
        # partition只把最高n个划到尾部，不完整排序；这里后续只取均值，所以不需要排序。
        diff=float(top.mean()-vals.mean());std=float(vals.std())
        # 高分均值比整段均值高且标准差满足阈值时停止分段；达到深度5或只剩1点也停止。
        # scores前面已缩放到0..1，std>-100通常成立；这仍是保留的原条件，不擅自简化。
        if (diff>t1 and std>t2) or depth>=max_depth or len(ids)<=1:
            out.append((ids,depth));return
        rec(ids[:len(ids)//2],depth+1)
        rec(ids[len(ids)//2:],depth+1)
    # 初始ids=0..n-1、depth=0。每次切片只划分候选位置，不读取Fine区间决定边界。
    rec(np.arange(len(scores)),0)
    return out


def aks_indices(visual_emb,sop_text_emb,K):
    """AKS adapted to SOP text + CLIP query, deterministic exact-K fill.

    Unlike original question-conditioned AKS, query here is all SOP steps mean;
    additional min-time-gap fill enforces K without seeing QA gold.
    """
    # all_indices传入CLIP图像向量(n,512)和SOP查询(512,)，不是题目或Fine标签。
    # SOP查询由vision.encode_clip逐步骤编码、归一化、平均再归一化，属于本地SOP适配版。
    x=np.asarray(visual_emb,np.float32);q=np.asarray(sop_text_emb,np.float32).reshape(-1)
    # x@q得到每张图与查询的点积（一维n个）；除两者长度得到余弦相似度，1e-12避免除零。
    score=x@q/(max(np.linalg.norm(q),1e-12)*np.maximum(np.linalg.norm(x,axis=1),1e-12))
    span=float(score.max()-score.min())
    # 减最小值、除极差，把本视频相似度缩到0..1；span过小时用下界避免除零。
    score=(score-score.min())/max(span,1e-12)
    n=len(score);initial=[]
    for ids,depth in aks_segments(score,K):
        # depth越大，单段初选quota越小。至少取1个；这一轮各段初选合计可能不恰好等于K。
        quota=max(1,int(K/2**depth))
        initial.extend(ids[np.argsort(-score[ids],kind='stable')[:quota]].tolist())
    # argsort(-score)是按分数从高到低排列下标；stable保留同分时的原顺序。
    # 全部初选再按(-分数,下标)排序，去重并最多保留K个，因此不会因多个片段超额。
    selected=[];seen=set()
    for i in sorted(initial,key=lambda i:(-score[i],i)):
        if i not in seen and len(selected)<K:seen.add(i);selected.append(i)
    while len(selected)<K:
        # 不足K时才补选：rem为未选下标；广播相减得到“剩余点×已选点”的下标距离矩阵。
        # 对axis=1取min得到每个剩余点离已选集合的最近距离；除n/K形成相对间隔。
        rem=np.asarray([i for i in range(n) if i not in seen])
        dist=np.min(np.abs(rem[:,None]-np.asarray(selected)[None,:]),axis=1)/max(1,n/K) if selected else np.ones(len(rem))
        # 注意这里用的是均匀网格下标距离，不是直接用秒；相似度之外加一点时间分散奖励。
        # .16是当前适配固定补选权重，优化来源未核实，不是CLIP训练出的参数。
        val=score[rem]+.16*np.minimum(1.,dist)
        i=int(rem[np.argmax(val)]);seen.add(i);selected.append(i)
    # 每轮补一个尚未使用点，直到K；返回下标给all_indices，不在本函数读Gold或计算覆盖率。
    return _check_indices(selected,n,K,'AKS_SOP_CLIP')


def all_indices(vid, times, K, coarse, hist=None, clip_image=None, clip_text=None):
    """Returns each applicable method as sorted integer positions within shared grid."""
    # 这是choose_frames Ctrl+单击进入的总入口；times为4K网格，其他数组行号必须与它对齐。
    # P08实际为times(160,)、hist(160,40)、clip_image(160,512)、clip_text(512,)、K=40。
    # 返回found={方法名:K个下标}，回到choose_frames后用times[ix]变成请求秒数。
    # 然后snap_to_mp4→冻结→evaluate.py统一评价；这里不直接生成覆盖率。
    n=len(times);seed=SEED+sum(map(ord,vid))
    # ord把视频ID各字符转整数后求和，给每个视频固定偏移；这是历史种子规则，不是字符串hash。
    found={
      'UNIFORM_FULL_VIDEO':uniform_indices(n,K),
      'RANDOM_K':random_indices(n,K,seed),
      'INVERSE_EXP_SOP_STAGE':inverse_exp_indices(times,coarse,K,seed),
    }
    if hist is not None:
        # HSV只进入这两种颜色方法；同一hist分别做KMeans代表选择和SVD/Logdet选择。
        found['VSUMM_HIST_KMEANS']=vsumm_indices(hist,K)
        found['MAXVOL_HIST_DOPT']=maxvol_indices(hist,K)
    if clip_image is not None and clip_text is not None:
        # 前两种用CLIP图像，AKS还用SOP查询；当前入口要求两者都存在才进入这个分支。
        found['MAXINFO_CLIP_MAXVOL_ADAPT']=maxvol_indices(clip_image,K)
        found['MAXINFO_CHUNK_CLIP_ADAPT']=maxinfo_chunk_indices(clip_image,K)
        found['AKS_SOP_CLIP_ADAPT']=aks_indices(clip_image,clip_text,K)
    # 名称AKS_SOP_CLIP_ADAPT在choose_frames中改成正式长名称；选点数组本身没有被修改。
    # 同DEV、同K不等于同信息：Lite另有Hand/Object/coarse Oracle和TRAIN Fine监督。
    return found


def uniform_pool_indices(times, K):
    """候选秒数→到K个均匀目标的最近未用下标；保留历史末候选端点规则。"""
    # choose_frames直接调用；输入是Lite候选时间，不是RGB的4K网格。P08有1110个Lite候选。
    # 用最后一个候选时间ts[-1]作为均匀目标的端点；不能换成视频duration，否则改变历史规则。
    ts = np.asarray(times, float)
    unused = np.ones(len(ts), bool)
    chosen = []
    for goal in np.linspace(ts[-1]/(2*K), ts[-1]-ts[-1]/(2*K), K):
        # 每个goal是均匀目标秒数；distance为所有候选与它的绝对秒差。
        distance = abs(ts-goal)
        distance[~unused] = np.inf
        # 已用点的距离设正无穷，argmin便只能取最近的未用候选；每轮标记后不再选它。
        i = int(np.argmin(distance))
        unused[i] = False
        chosen.append(i)
    # 返回K个候选下标给choose_frames，再用pool_times[chosen]取秒数并走统一snap/冻结/评价。
    return chosen
