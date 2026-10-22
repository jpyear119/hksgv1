"""真实RGB→JPEG/HSV40/CLIP512；或核验既有缓存。所有模型仅从本地读取。"""
from __future__ import annotations
# 这份文件负责“真实视频/缓存→数值特征”，主要由run1019.choose_frames()调用get_features()。
# 它没有main或命令行解析，单独python vision.py不会训练、选帧或评价；入口在run1019.py。
# 顺着读：找FFmpeg→读视频时间信息→解码JPEG→核验帧→HSV/CLIP→把h、X、q交回choose_frames。
import hashlib, json, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
import numpy as np
from PIL import Image, ImageStat, ImageFilter
from data import read_json, write_json_new, sha256

MODEL_ID = 'openai/clip-vit-base-patch32'
# Stop/sha/write是已有函数的别名，沿用统一报错、SHA256和拒绝覆盖的JSON写入行为。
Stop = RuntimeError
sha = sha256
write = write_json_new
DECODER = {'RGB_width': 1024, 'RGB_jpeg_q': 2, 'RGB_timebase': '1:1000000'}
# 1024是缩放后JPEG宽度；2是FFmpeg的JPEG质量参数；1:1000000表示编码时间戳的单位为微秒。
# 时间基不是每秒采样一百万帧；真正采样时刻由baselines.grid()产生的times决定。
# 这些是历史继承的人工固定解码参数，优化来源未核实，不能为改善覆盖率改变它们。


def config():
    """返回冻结的JPEG参数；不使用Gold。"""
    # decode_sequence和image_properties调用，返回同一个DECODER字典，避免各处使用不同图片规格。
    return DECODER


def ffmpeg():
    # get_features/解码过程调用；不传参数，返回已经安装的FFmpeg可执行文件路径字符串。
    # imageio_ffmpeg只在这里提供本地程序位置；本函数没有下载或安装操作。
    import imageio_ffmpeg
    f=imageio_ffmpeg.get_ffmpeg_exe()
    if not Path(f).is_file():raise Stop('FFMPEG_EXECUTABLE_NOT_FOUND')
    return f


def probe(src,ff):
    # get_features的raw分支调用：src是本地MP4路径，ff是ffmpeg()返回的程序路径。
    # subprocess.run启动外部FFmpeg并等待它结束；真正读容器信息的是FFmpeg，不是Python自己解码。
    # 这里没有指定输出文件，主要解析stderr里的时长、起始时间和fps，不以返回码0作为探测条件。
    p=subprocess.run([ff,'-hide_banner','-nostdin','-i',str(src)],capture_output=True,text=True,encoding='utf-8',errors='replace')
    s=p.stderr;m=re.search(r'Duration:\s*(\d+):(\d+):([0-9.]+),\s*start:\s*(-?[0-9.]+)',s)
    # 正则表达式把“时:分:秒、start”分组取出；fps是名义帧率，实际帧时间仍要看PTS/官方CSV。
    fps=re.search(r'Video:[^\n]*?\b([0-9]+(?:\.[0-9]+)?)\s+fps\b',s)
    if not m or not fps:raise Stop('REAL_MP4_PROBE_INCOMPLETE '+str(src)+' '+s[-1000:])
    # 返回字典给get_features的meta；format_start_s用于在绝对PTS与视频内相对秒数之间换算。
    # 实验预算用DEV便携输入的duration，不拿这个容器显示时长替换原预算。
    return {'duration_s':int(m[1])*3600+int(m[2])*60+float(m[3]),'format_start_s':float(m[4]),'fps':float(fps[1]),'stderr':s}


def packet_rows(path):
    # decode_sequence调用：path是FFmpeg刚生成的packet.framehash文本，不是RGB图片。
    # PTS是显示时间戳：一个整数配合time_base（每个刻度多少秒）才表示真实时间。
    s=Path(path).read_text(encoding='utf-8');m=re.search(r'#tb\s+0:\s*(\d+)/(\d+)',s)
    if not m:raise Stop('PACKET_TIMEBASE_MISSING')
    tb=Fraction(int(m[1]),int(m[2]));rows=[]
    # Fraction保存分子/分母，先按有理数乘PTS再转float，避免先把时间基粗略取整。
    for line in s.splitlines():
        if not line.strip() or line.startswith('#'):continue
        p=[v.strip() for v in line.split(',')]
        # 每个有效记录里有流编号、PTS、编码包字节数、SHA256；size不是图片的宽×高。
        if len(p)<6 or int(p[0])!=0:raise Stop('INVALID_FRAMEHASH_RECORD')
        rows.append({'pts':int(p[2]),'time_base':str(tb),'packet_pts_s':float(int(p[2])*tb),
                     'size':int(p[4]),'packet_sha256':p[5]})
    # 返回字典列表给decode_sequence的packets，用来把JPEG文件字节与同一个编码包绑定。
    return rows


def image_properties(path):
    # decode_sequence在拿到JPEG后调用：path为刚写出的图片，返回尺寸、像素SHA和简单诊断值。
    # 先verify检查文件能否完整解码，再重新open实际读像素；with结束时关闭图片文件。
    with Image.open(path) as im:im.verify()
    with Image.open(path) as im:
        rgb=im.convert('RGB');w,h=rgb.size
        if w!=config()['RGB_width'] or h<180:raise Stop('DECODE_IMAGE_DIMENSION_WRONG')
        gray=rgb.resize((256,144)).convert('L');stats=ImageStat.Stat(gray)
        # RGB是红绿蓝三通道；L为灰度。缩成256×144后取均值、标准差和边缘强度作质量诊断。
        # 高度180、边缘4、灰度标准差8等是已有固定检查阈值，优化来源未核实，不是动作真值。
        edge=ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0]
        # decoded_RGB_sha256针对解码后的像素；jpeg_sha256针对压缩文件，两种SHA的对象不同。
        # 诊断值被存入回执，不参与all_indices的选帧分数，也不判断语义动作是否正确。
        return {'width':w,'height':h,'decoded_RGB_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
         'gray_mean':stats.mean[0],'gray_stddev':stats.stddev[0],'edge_strength_proxy':edge,
         'low_detail_review_flag':bool(edge<4 or stats.stddev[0]<8),
         'quality_is_diagnostic_not_semantic_ground_truth':True}


def decode_sequence(ff,src,targets,outdir,start_s,fast):
    """Fast: one input seek. Reference: one sequential pass for all sentinel targets."""
    # 谁调用：get_features(raw)里面的decode_one；顺序参考核验也可用fast=False调用它。
    # ff/src是程序和MP4路径；targets为请求的“视频内秒数”列表；outdir放本次JPEG/日志。
    # start_s来自probe的format_start_s；fast=True只接受一个target，False允许顺序处理多个。
    # 返回rows字典列表；raw单帧调用取[0]赋给row，再加image_path，最终交给HSV/CLIP读取。
    outdir=Path(outdir)
    if (outdir/'ffmpeg.stderr.txt').exists():
        # 这是原来的重试保留规则：已有未接受尝试另存，再用原目录继续。正式流程应给全新输出目录。
        previous=outdir.with_name(outdir.name+'_previous_unaccepted_attempt_'+str(time.time_ns()))
        outdir.rename(previous)
    outdir.mkdir(parents=True,exist_ok=True);targets=sorted(set(float(t) for t in targets))
    if fast and len(targets)!=1:raise Stop('FAST_DECODE_MUST_REQUEST_SINGLE_TARGET')
    absolute=[t+start_s for t in targets]
    # 请求t是相对视频开始的秒数；FFmpeg保留原始PTS，所以过滤时要加容器start_s。
    # 每个pieces在时间>=目标且前一次保留帧仍早于目标时才通过；这样取目标后第一帧。
    pieces=[f'gte(t\\,{t:.9f})*if(isnan(prev_selected_t)\\,1\\,lt(prev_selected_t\\,{t:.9f}))' for t in absolute]
    vf='select='+('+'.join(pieces))+f',trim=end_frame={len(targets)},scale={config()["RGB_width"]}:-2,format=yuvj420p,showinfo'
    # trim限制输出张数；scale固定宽1024并按比例取偶数高度；showinfo记录实际保留帧的PTS。
    args=[ff,'-hide_banner','-nostdin','-loglevel','info','-y','-threads','1','-filter_threads','1','-copyts']
    if fast:args+=['-ss',f'{targets[0]:.9f}','-accurate_seek']
    # -ss先定位到目标附近，-copyts保留源时间；select仍按PTS筛帧，不能只把-ss当作准确帧证明。
    args+=['-i',str(Path(src).resolve()),'-map','0:v:0','-an','-sn','-dn','-vf',vf,
           '-fps_mode','passthrough','-frames:v',str(len(targets)),'-c:v','mjpeg','-threads:v','1',
           '-q:v',str(config()['RGB_jpeg_q']),'-enc_time_base',config()['RGB_timebase'],'-f','tee',
           '[f=image2]frame_%03d.jpg|[f=framehash:hash=sha256]packet.framehash']
    # MJPEG编码同一份输出包，tee同时写JPEG和framehash，避免“图片来自一帧，时间来自另一帧”。
    # 采样成可校验JPEG后，HSV和CLIP读取同一批图像，也能保存、重放和逐张检查实际输入。
    st=time.monotonic();p=subprocess.run(args,cwd=outdir,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (outdir/'ffmpeg.stderr.txt').write_text(p.stderr,encoding='utf-8')
    write(outdir/'command.json',{'args':args,'mode':'FAST_INPUT_SEEK' if fast else 'SEQUENTIAL_REFERENCE'})
    if p.returncode:raise Stop('FFMPEG_DECODE_FAILED '+p.stderr[-2000:])
    # Ctrl+单击packet_rows看编码包的PTS/SHA；Alt+←回来再对照showinfo的保留帧日志。
    packets=packet_rows(outdir/'packet.framehash')
    tb=re.search(r'config in time_base:\s*(\d+)/(\d+)',p.stderr)
    logs=re.findall(r'\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:\s*([-+0-9.eE]+)',p.stderr)
    if not tb or len(logs)!=len(targets) or len(packets)!=len(targets):
        raise Stop('ONE_TO_ONE_RETAINED_FRAME_PACKET_NOT_ESTABLISHED '+str({'logs':logs,'packets':packets,'targets':targets}))
    itb=Fraction(int(tb[1]),int(tb[2]));rows=[]
    for j,(packet,log,t) in enumerate(zip(packets,logs,targets),1):
        # zip把“编码包、保留帧日志、请求秒数”一一配对；j从1开始对应frame_001.jpg。
        fp=outdir/f'frame_{j:03d}.jpg'
        if not fp.is_file() or sha(fp)!=packet['packet_sha256'] or fp.stat().st_size!=packet['size']:
            raise Stop('JPEG_BYTES_NOT_THE_HASHED_PACKET')
        inputpts=float(int(log[0])*itb)
        # 一次核JPEG文件SHA/字节数，一次核保留帧与编码包的PTS；不是仅凭文件名认定帧时间。
        # 1.1e-6是已有PTS比较容差（约微秒），不等于评价的±2秒，也不用于修改采样点。
        if abs(inputpts-packet['packet_pts_s'])>1.1e-6:raise Stop('ENCODED_PACKET_PTS_NOT_RETAINED_FRAME_PTS')
        actual=inputpts-start_s
        # actual换回视频内秒数，并检查它没有早于请求；返回值会写进ORDERED_JPEGS清单。
        if actual<t-1e-8:raise Stop('SELECTED_FRAME_PRECEDES_REQUEST')
        rows.append({**packet,**image_properties(fp),'requested_s':t,'actual_relative_s':actual,
         'retained_frame_pts':int(log[0]),'retained_frame_timebase':str(itb),
         'jpeg_sha256':sha(fp),'elapsed_total_s':round(time.monotonic()-st,4),
         'source_absolute_PTS_preserved':True,'jpeg_filename':fp.name})
    return rows


def color_histogram(image):
    """HSV 16+12+12 = 40 dimensional feature; RGB decodes are not re-trained."""
    # histograms调用；输入是Pillow图片，输出一维float32的40个数，不是训练出的特征模型。
    # HSV的H是色相、S是饱和度、V是明度；把像素按颜色区间计数可描述一帧的整体颜色分布。
    from PIL import Image
    x=image.convert('RGB').resize((96,54),Image.Resampling.BILINEAR)
    # 统一缩成96×54，用双线性插值；reshape(-1,3)把像素排成“像素数×三个通道”。
    # 96×54、16/12/12个区间是历史人工固定设计，优化来源未核实，不因DEV评价重新设置。
    hsv=np.asarray(x.convert('HSV'),dtype=np.uint8).reshape(-1,3)
    feats=[]
    for c,n_bins in ((0,16),(1,12),(2,12)):
        # uint8通道值为0..255；乘区间数再整除256，把每个值放进一个bin（计数区间）。
        # bincount统计各区间有多少像素；三个通道分别16、12、12项，拼起来共40维。
        h=np.bincount((hsv[:,c].astype(int)*n_bins)//256,minlength=n_bins).astype(np.float32)
        h/=max(1,float(h.sum()));feats.extend(h.tolist())
        # 每个通道除本通道像素总数，变成比例；因此各通道分别和为1，整条40维通常和为3。
    return np.asarray(feats,np.float32)


def histograms(jpeg_files):
    # get_features(raw)调用；jpeg_files按times网格顺序排列，不能打乱，否则行号与时间失配。
    # 每张图调用color_histogram返回(40,)，stack沿新轴堆成(n,40)，返回给new_h。
    from PIL import Image
    results=[]
    for f in jpeg_files:
        with Image.open(f) as im:results.append(color_histogram(im))
    return np.stack(results)


def snap_to_mp4(requests, clock):
    """Same first-frame-at-or-after-request mapping used by the V4 decoder.

    Resolve collisions forward, then backward at the final frame; never use GT.
    Preserve original requests, frame indices, adjustments and mapping errors.
    """
    # 谁调用：run1019.choose_frames的每种方法都走这里，不能各写一套时间映射。
    # requests是K个请求秒数；clock是官方CSV第一列纳秒/1e9得到的一维递增帧时间数组。
    # 返回mapped字典：原请求、零起点官方帧号、实际秒数、误差和冲突记录，交给freeze_selections。
    requests = list(map(float, requests))
    assert len(requests) == len(set(requests)) and np.isfinite(requests).all()
    used = set(); indices = []; conflicts = []
    for t in requests:
        # searchsorted默认left：寻找第一个>=t-1e-9的位置；这里的1e-9是保留的历史容差。
        # 超出最后一帧时截到最后下标。注意返回j是帧号，不是秒数，也不是4K候选下标。
        j = min(int(np.searchsorted(clock, t-1e-9)), len(clock)-1)
        original = j
        while j in used and j+1 < len(clock):
            j += 1
        if j in used:
            # 已被别的请求占用时，先向后找；若到末尾仍冲突，再回到original向前找。
            j = original
            while j in used and j > 0:
                j -= 1
        assert j not in used
        used.add(j); indices.append(j)
        if j != original:
            conflicts.append({'requested_s': t, 'initial_frame_index': original,
                              'resolved_frame_index': j})
    actual = [float(clock[j]) for j in indices]
    # P08的网格下标2请求18.50799967453125秒，映射为官方帧556、18.53333333333333秒。
    # mapping_error_s=实际-请求；冻结保存两者，不能把请求秒数直接宣称为实际MP4帧时间。
    return {'requested_times_s': requests, 'selected_times_s': actual,
            'official_frame_indices_zero_based': indices,
            'mapping_error_s': [a-t for a, t in zip(actual, requests)],
            'collision_resolution': conflicts}


def encode_clip(rows, steps, snapshot):
    """JPEG和SOP步骤→FP32单位向量→逐步骤平均后归一化；冻结CPU/慢处理器/批量8。"""
    # get_features(raw)调用：rows给出有序JPEG路径，steps是DEV中对应recipe的SOP步骤字典列表。
    # SOP是操作步骤文本；snapshot为paths.json指定的本地CLIP模型目录，不是新训练输出。
    # CLIP是预训练的图像/文本编码模型，把两类输入映射到可比较的向量空间；此ViT-B/32输出512维。
    # 512来自所选预训练模型的投影结构，不是本轮手工调出的维度；这里不fit、不读取Fine。
    import torch
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(4)
    # CPU线程4、批量8、use_fast=False均继承历史特征流程，优化来源未核实；不是HGB参数。
    # local_files_only=True只读本地权重/处理器；eval切到推断状态，inference_mode不记录训练梯度。
    model = CLIPModel.from_pretrained(str(snapshot), local_files_only=True).eval()
    processor = CLIPProcessor.from_pretrained(str(snapshot), local_files_only=True, use_fast=False)
    texts = [s['text'] for s in steps if s.get('text')] or ['cooking procedure']
    # 列表推导式按原顺序取非空text；没有文本时用固定回退句，不从Gold补出步骤。
    X, Q = [], []
    with torch.inference_mode():
        for i in range(0, len(rows), 8):
            # 每次取最多8张JPEG，processor做模型要求的图像预处理，返回PyTorch张量。
            ims = []
            for row in rows[i:i+8]:
                with Image.open(row['image_path']) as im:
                    ims.append(im.convert('RGB'))
            ins = processor(images=ims, return_tensors='pt')
            X.append(model.get_image_features(pixel_values=ins['pixel_values']).float().numpy())
            # 一批返回(b,512)；float是FP32，numpy转换成数组，先放进X列表，后面再合并各批。
        for i in range(0, len(texts), 8):
            # SOP逐步骤编码；padding补齐批内长度，truncation沿用模型的长度截断规则。
            ins = processor(text=texts[i:i+8], padding=True, truncation=True, return_tensors='pt')
            Q.append(model.get_text_features(**ins).float().numpy())
    X = np.concatenate(X).astype(np.float32)
    # 合并后X为(n,512)，Q为(步骤数,512)；不要把X（CLIP图像）与prepare.frames的X141混为一谈。
    Q = np.concatenate(Q).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    # 每行除欧氏长度，保留方向，使后续点积可解释成余弦相似度；下界只防止零向量除零。
    Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    q = Q.mean(axis=0)
    # 先逐步骤归一化Q，再跨步骤取均值，最后把均值q再归一化；不是先把所有SOP拼成长句。
    q /= max(float(np.linalg.norm(q)), 1e-12)
    # 返回图像矩阵和一个SOP查询向量给new_X/new_q。P08分别为(160,512)和(512,)。
    return X, q


def get_features(video_id, times, steps, clock, paths, out, mode='cached'):
    """核验缓存身份/SOP/形状；raw模式实际解码MP4并重新计算，写出逐值比较证据。"""
    # choose_frames在每个视频调用，返回h、X、q后立即交给baselines.all_indices。
    # times为4K个候选秒数，steps来自本视频coarse关联的SOP，clock为官方全部帧时间。
    # paths提供缓存/MP4/权重路径，out放本次核验回执；本函数不选最终K帧，也不评价Fine。
    folder = paths['features'] / video_id
    receipt = read_json(folder / 'FEATURE_RECEIPT.json')
    cache = folder / 'SHARED_HSV40_CLIP512.npz'
    ordered = folder / 'ORDERED_JPEGS.json'
    # cached和raw都会先核可信NPZ与有序JPEG清单的SHA、SOP身份及数组形状，作为比较基准。
    # NPZ中histograms→h(n,40)、frame_features→X(n,512)、sop_features→q(512,)。
    if sha256(cache) != receipt['feature_npz_sha256'] or sha256(ordered) != receipt['ordered_JPEG_manifest_sha256']:
        raise ValueError('VISUAL_CACHE_SHA_CHANGED ' + video_id)
    if receipt['SOP_steps'] != steps or receipt['Fine_access'] is not False:
        raise ValueError('VISUAL_CACHE_SOP_OR_PERMISSION_CHANGED ' + video_id)
    if receipt['CLIP_model_id'] != MODEL_ID:
        raise ValueError('CLIP_MODEL_ID_CHANGED')
    with np.load(cache) as z:
        # 生成器逐个按键读取并copy，让数组在退出np.load的with之后仍独立可用。
        h, X, q = (z[k].copy() for k in ['histograms', 'frame_features', 'sop_features'])
    if h.shape != (len(times), 40) or X.shape != (len(times), 512) or q.shape != (512,):
        raise ValueError('VISUAL_SHAPE_MISMATCH')
    if not all(np.isfinite(a).all() for a in (h, X, q)):
        raise ValueError('VISUAL_NONFINITE')
    rows = read_json(ordered)
    # zip把清单与times配对；检查请求和实际秒数对得上官方clock，并检查清单长度。
    # cached这里只读NPZ和清单，不重新解码JPEG或重新跑CLIP，不能将其称为Raw重提取。
    if any(abs(r['requested_s']-t) > 1e-9 or
           abs(r['actual_relative_s']-clock[min(int(np.searchsorted(clock, t-1e-9)), len(clock)-1)]) > 1e-6
           for r, t in zip(rows, times)) or len(rows) != len(times):
        raise ValueError('VISUAL_FRAME_GRID_CHANGED')
    proof = {'video_id': video_id, 'mode': 'CACHED_FEATURE_REPLAY',
             'npz_sha256': sha256(cache), 'source_receipt_sha256': sha256(folder / 'FEATURE_RECEIPT.json'),
             'CLIP_model_id': MODEL_ID, 'CLIP_weight_sha256': receipt['CLIP_weight_sha256'],
             'HSV_shape': list(h.shape), 'CLIP_shape': list(X.shape), 'Fine_access': False}
    if mode == 'raw':
        # raw额外核本地MP4的SHA与官方MD5，随后真正读取视频并重算特征；没有用旧向量替代新输出。
        source = read_json(paths['video_sources'])[video_id]
        mp4 = paths['root'] / source['mp4']
        if sha256(mp4) != source['mp4_sha256']:
            raise ValueError('MP4_SHA_CHANGED')
        with mp4.open('rb') as f:
            if hashlib.file_digest(f, 'md5').hexdigest() != source['official_md5']:
                raise ValueError('MP4_OFFICIAL_MD5_MISMATCH')
        meta = probe(mp4, ffmpeg())
        # Ctrl+单击probe看start_s；返回后ffprobe列出全部packet PTS，排序并减start得到相对秒数。
        probe_args = [str(paths['ffprobe']), '-v', 'error', '-select_streams', 'v:0',
                      '-show_packets', '-show_entries', 'packet=pts_time', '-of', 'csv=p=0', str(mp4)]
        p = subprocess.run(probe_args, capture_output=True, text=True, encoding='utf-8')
        if p.returncode:
            raise RuntimeError(p.stderr)
        pts = np.sort(np.array([float(line.split(',')[0]) for line in p.stdout.splitlines() if line.strip()])) - meta['format_start_s']
        if len(pts) != len(clock) or np.max(abs(pts-clock)) > 1.1e-6:
            raise ValueError('MP4_PTS_CLOCK_MISMATCH')
        raw_folder = Path(out) / 'raw_features' / video_id
        raw_folder.mkdir(parents=True, exist_ok=False)
        def decode_one(item):
            # enumerate(times)提供(网格下标i,请求秒数t)；i只用来命名独立目录，避免并发写同一文件。
            i, t = item
            frame_dir = raw_folder / f'{i:04d}'
            row = decode_sequence(ffmpeg(), mp4, [float(t)], frame_dir, meta['format_start_s'], True)[0]
            # Ctrl+单击decode_sequence看JPEG/PTS绑定；返回后补image_path，供Pillow读取同一张JPEG。
            row['image_path'] = str(frame_dir / row['jpeg_filename'])
            return row
        rows = []
        # 每个请求仍调用原单帧解码器；map保持网格顺序，不改变JPEG、PTS或选点。
        with ThreadPoolExecutor(max_workers=8) as pool:
            # 8个独立解码请求并发；pool.map按输入顺序交回结果，所以rows仍与times逐行对应。
            for row in pool.map(decode_one, enumerate(times)):
                rows.append(row)
                if len(rows) % 50 == 0:
                    print('RAW_RGB', video_id, len(rows), '/', len(times), flush=True)
        new_h = histograms([r['image_path'] for r in rows])
        # Ctrl+单击histograms→color_histogram，Alt+←回来；new_h为(n,40)。
        # 下一行进encode_clip后回到这里，得到new_X(n,512)、new_q(512,)。
        new_X, new_q = encode_clip(rows, steps, paths['clip_snapshot'])
        comparison = {key: {'exact': bool(np.array_equal(old, new)),
                             'max_abs_error': float(np.max(abs(old-new)))}
                      for key, old, new in [('HSV40', h, new_h), ('CLIP512', X, new_X), ('SOP512', q, new_q)]}
        # 字典推导式记录“逐值完全相等”和最大绝对差；这里只比较，不用结果去舍入/替换特征。
        # 已有全Raw结果：23视频/3728张JPEG一致，但三份早期CLIP缓存有微差，MaxInfo共5/690项不同。
        # Raw严格等价尚未通过；cached PASS不能写成Raw PASS，具体原编码环境仍有身份记录限制。
        original = read_json(ordered)
        proof.update(mode='RGB_RAW_REPLAY', MP4_sha256=source['mp4_sha256'],
                     MP4_official_md5=source['official_md5'], PTS_max_error_s=float(np.max(abs(pts-clock))),
                     comparison=comparison,
                     same_JPEG_count=sum(a['jpeg_sha256'] == b['jpeg_sha256'] for a, b in zip(rows, original)),
                     frame_count=len(rows), ffmpeg_sha256=sha256(ffmpeg()),
                     ffprobe_command=probe_args, ffprobe_returncode=p.returncode)
        h, X, q = new_h, new_X, new_q
        # raw返回的是刚算出的new_*，同时保留比较证据；不能为追历史分数换回旧h/X/q。
        np.savez_compressed(raw_folder / 'SHARED_HSV40_CLIP512.npz', histograms=h, frame_features=X, sop_features=q)
        write_json_new(raw_folder / 'ORDERED_JPEGS.json', rows)
    write_json_new(Path(out) / 'feature_checks' / (video_id + '.json'), proof)
    # cached返回核验后的旧缓存，raw返回本轮新特征；返回choose_frames的同名h、X、q继续选点。
    return h, X, q

