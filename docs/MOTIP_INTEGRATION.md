# StreamPETR + MOTIP Integration

SparseBEV+MOTIP 코드를 StreamPETR detector로 옮기기 위한 통합 작업 기록.

---

## Why StreamPETR

기존 SparseBEV+MOTIP 구현은 학습 시 CPU bottleneck이 심함:

```
clip_len × num_cameras × (1 + sweeps_num) × per_image_cost
   4     ×       6     ×     (1 + 7)      = 192 images per sample
```

- SparseBEV: keyframe당 7 sweeps을 디스크에서 로드 → temporal model 입력
- MOTIP: clip 단위 학습 (clip_len=4)
- 두 시간축이 곱해져서 polynomial하게 폭발

→ StreamPETR로 갈아타면 sweeps 없이 query memory propagation으로 시간 정보 처리. **per-sample I/O가 1/8 수준** (192 → 24).

---

## 환경 호환성

### 사용 환경

| 항목 | 버전 |
|------|------|
| Python | 3.8 |
| CUDA | 11.8 |
| PyTorch | 2.0.0+cu118 |
| mmcv-full | 1.7.0 (custom build at /tmp/mmcv) |
| mmdet | 2.28.2 |
| mmdet3d | 1.0.0rc6 |
| GPU | RTX 4090 (sm_89, PTX JIT fallback) |

`sparsebev` conda env 그대로 사용 (`/data4/minjae/miniforge3/envs/sparsebev`).

### 환경 변수

```bash
LD_LIBRARY_PATH=/data4/minjae/miniforge3/envs/sparsebev/lib:$LD_LIBRARY_PATH
PYTHONPATH=/data4/minjae/workspace/StreamPETR:$PYTHONPATH
```

`LD_LIBRARY_PATH`는 conda env의 새 libstdc++을 시스템 것보다 우선시키기 위함 (GLIBCXX_3.4.29 필요).

---

## 환경 셋업 수정사항

### 1. mmdetection3d configs symlink

StreamPETR base config가 `../../../mmdetection3d/configs/_base_/...`를 참조하지만 사용자가 mmdetection3d를 clone하지 않음. mmdet3d 패키지의 `.mim/configs`를 symlink로 연결:

```bash
ln -s /data4/minjae/miniforge3/envs/sparsebev/lib/python3.8/site-packages/mmdet3d/.mim \
      /data4/minjae/workspace/StreamPETR/mmdetection3d
```

### 2. flash-attn 우회 (옵션 2-A)

`flash_attn` 라이브러리는 설치하지 않음. 대신 import를 try/except로 감쌈.

**파일**: `projects/mmdet3d_plugin/models/utils/attention.py`
**라인**: 21-30 (원본의 21-22 두 줄을 try 블록으로 확장)

```python
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis
    _HAS_FLASH = True
except ImportError:
    _HAS_FLASH = False
    flash_attn_unpadded_kvpacked_func = None
    unpad_input = pad_input = index_first_axis = None
```

이렇게 해두면 `attention.py`가 import는 통과하지만 `FlashAttention`/`FlashMHA` 클래스를 instantiate하면 NameError. R50 config에서 `PETRMultiheadFlashAttention` → `PETRMultiheadAttention`으로 교체해서 instantiate를 피함.

근거: maintainer 공식 답변 (https://github.com/exiawsh/StreamPETR/issues/23). flash-attn은 "optional"이지만 해제하려면 import도 같이 손봐야 함.

### 3. 디버그 leftover 제거

**파일**: `projects/mmdet3d_plugin/datasets/samplers/group_sampler.py`
**라인**: 18

```python
# from IPython import embed  # disabled: debug-only, IPython not installed
```

---

## Non-flash override config

기존 60e R50 config의 attention만 교체한 override config:

`projects/configs/StreamPETR/stream_petr_r50_noflash_704_bs2_seq_428q_nui_60e.py`:

```python
_base_ = ['./stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.py']

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(type='MultiheadAttention',
                             embed_dims=256, num_heads=8, dropout=0.1),
                        dict(type='PETRMultiheadAttention',  # ← was PETRMultiheadFlashAttention
                             embed_dims=256, num_heads=8, dropout=0.1),
                    ],
                ),
            ),
        ),
    ),
)
```

`PETRMultiheadAttention`은 동일 layout (in_proj_weight, out_proj.*)을 사용해서 60e 체크포인트가 그대로 load됨.

---

## 검증 결과

### Forward pass on val[0]

```
Total params: 37.26 M
Checkpoint loaded.
all_cls_scores: (6, 1, 428, 10)
all_bbox_preds: (6, 1, 428, 10)
top-5 scores: [0.94, 0.84, 0.28, 0.26, 0.23]
```

- non-flash 변환이 수치적으로 정상 (top score 합리적)
- 6 decoder layer, 428 query (300 base + 128 propagated), 10 class
- num_propagated=128로 인해 query 수가 num_query보다 큰 게 정상

---

## StreamPETR 시간 메커니즘 (MOTIP과의 관계)

### Head 내부 메모리

`StreamPETRHead`는 자체 memory bank를 운영:

| 변수 | 역할 |
|------|------|
| `memory_embedding` | 과거 query feature [B, 512, C] |
| `memory_reference_point` | 과거 query 위치 [B, 512, 3] |
| `memory_egopose` | 과거 query의 ego pose [B, 512, 4, 4] |
| `memory_velo` | 과거 query 속도 |
| `memory_timestamp` | 시간 정보 |

매 frame:
1. **`pre_update_memory`**: 이전 메모리를 현재 ego pose 좌표계로 변환 + scene boundary 시 reset (`prev_exists=0`)
2. **`temporal_alignment`**: 메모리 앞 128개를 새 query 300개와 concat → 428개 query
3. Decoder forward → cls/bbox 예측
4. **`post_update_memory`**: 현재 frame의 top-128 query를 다시 메모리에 push (`.detach()` — gradient 안 흐름)

### MOTIP과의 컨셉 비교

| 항목 | StreamPETR memory | MOTIP tracklet buffer |
|------|-------------------|-----------------------|
| 저장 단위 | top-128 query (score 기준) | matched detection (Hungarian 기준) |
| 사용 목적 | 다음 frame **detection** 품질 | frame 간 **ID 매칭** |
| 길이 | memory_len=512 | context_len=5 |
| Identity 추적 | ❌ implicit | ✅ explicit ID |
| Detach | ✅ | ✅ |
| 좌표 변환 | 매 frame `ego_pose` 기반 | 매 frame `lidar2global` 기반 |

**좌표 변환은 변수명만 다름**. StreamPETR의 `ego_pose`는 실제로는 `e2g @ l2e = lidar2global` (`nuscenes_dataset.py:176`). 동일 행렬, 동일 의미.

→ 두 시스템은 **충돌하지 않고 보완적**. 같은 query feature를 두 가지 목적으로 사용 가능.

---

## 통합 전략: A (MOTIP buffer를 별도 운영)

- StreamPETR head memory는 detection용으로 그대로 유지
- MOTIP은 자체 tracklet buffer 운영 (SparseBEV+MOTIP 코드 거의 100% 재사용)
- 두 시스템 독립 → 디버깅 쉬움

### Query 적용 범위: 1-B (전체 428 query 사용) — 1-A에서 변경됨

처음에는 결정 1-A (새 300개만) 로 가려 했으나, 다음 사실을 인지하면서 1-B로 변경:

- Hungarian matching의 매칭 수는 query 수가 아니라 **GT 수에 bound** (한 frame ~50개 GT). 즉 300이든 428이든 실제 학습 sample 수는 동일.
- 차이는 "어떤 query가 매칭되는가". 428을 쓰면 이전 frame에서 잘 track되고 있는 GT는 propagated query와 매칭됨 → MOTIP이 "propagated query에 대해 persistent ID 예측"을 학습. **이건 우리가 원하는 신호**.
- propagated query는 `head.memory_embedding`에서 `.detach()`되어 가져옴 + 현재 frame decoder가 한 번 더 cross-attention으로 update. 따라서 e2e 모드에서도 gradient는 현재 frame decoder까지 정상 flow (이전 frame까지는 끊어짐, StreamPETR의 의도된 설계).

→ **`outs['query_feat']` 전체 (1, 428, C)를 그대로 MOTIP에 사용**.

Hungarian matching은 검출기와 동일하게 428 전체에서 수행. 필터링/슬라이싱 없음.

---

## 작업 단계

각 단계 후 검증.

### Step 1: 데이터 준비 ✅

**새 파일**: `tools/prepare_motip_pkl.py` (전체 신규)

- SparseBEV pkl (`nuscenes_infos_*_sweep.pkl`)의 `instance_tokens` 필드를 StreamPETR pkl (`nuscenes2d_temporal_infos_*.pkl`)로 token 매칭으로 복사
- 사전 검증: token 28130/28130 일치, gt_boxes 내용까지 binary 일치 → 인덱스 그대로 복사 가능
- **결과**: train 28130/28130, val 6019/6019 모두 성공 (skipped/mismatch 0)
- 출력 파일 (원본 보존):
  - `data/nuscenes/nuscenes2d_temporal_infos_train_motip.pkl`
  - `data/nuscenes/nuscenes2d_temporal_infos_val_motip.pkl`

### Step 2: Dataset 클래스 수정 ✅

**파일**: `projects/mmdet3d_plugin/datasets/nuscenes_dataset.py`

**수정 1**: `get_data_info` 안에서 `gt_instance_ids`를 top-level로 노출
**라인**: 240-243 (기존 `input_dict['ann_info'] = annos` 다음에 추가)

```python
# MOTIP: expose gt_instance_ids at top-level so pipeline transforms
# can sync filtering without going through LoadAnnotations3D.
input_dict['gt_instance_ids'] = annos.get(
    'gt_instance_ids', torch.zeros(0, dtype=torch.long))
```

**수정 2**: `get_ann_info` 메서드 새로 override (부모 NuScenesDataset의 메서드 확장)
**라인**: 247-293 (새 메서드, 약 47줄)

- 부모 클래스의 ann_info에 `gt_instance_ids` 추가
- `_instance_token_map` (per-dataset-instance) 으로 string token → int ID 변환
- 부모와 동일한 valid_flag mask 적용 (인덱스 정렬 보존)
- instance_tokens 없으면 빈 텐서 반환 (downstream에서 안전하게 skip)

**검증**: sample 0/1 (같은 scene)이 동일한 첫 5개 ID `[0, 1, 2, 3, 4]` — frame 간 ID 일관성 OK

**주의**: DataLoader worker별, DDP rank별로 `_instance_token_map`이 독립적. 한 worker가 한 clip을 처리한다는 가정 하에서만 안전.

### Step 3: Pipeline transforms 추가 ✅

**파일**: `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py`

**수정 1**: import 추가
**라인**: 13-18 (기존 import 블록 확장)
```python
from mmdet3d.datasets.pipelines import ObjectRangeFilter, ObjectNameFilter
from mmcv.parallel import DataContainer as DC
```

**수정 2**: 헬퍼 함수 + 새 클래스 3개
**라인**: 22-86 (전체 신규, 약 65줄)

- `_sync_instance_ids(results, mask_np)` — line 22 — 마스크를 gt_instance_ids에 in-place로 적용
- `class ObjectRangeFilterWithIDs(ObjectRangeFilter)` — line 34 — range filter + ID 동기화 (mmdet3d 본가 상속)
- `class ObjectNameFilterWithIDs(ObjectNameFilter)` — line 57 — name filter + ID 동기화 (mmdet3d 본가 상속)
- `class WrapInstanceIDs` — line 71 — DataContainer로 wrap (variable-length tensor 처리)

mmdet3d 본가 클래스를 상속한 이유: DRY + 미래 mmdet3d 버그 픽스 자동 흡수.

**파일**: `projects/mmdet3d_plugin/datasets/pipelines/__init__.py`
**라인**: 1-9 (기존 import 블록에 3개 export 추가)

```python
from .transform_3d import(
    PadMultiViewImage,
    NormalizeMultiviewImage,
    ResizeCropFlipRotImage,
    GlobalRotScaleTransImage,
    ObjectRangeFilterWithIDs,    # ← MOTIP
    ObjectNameFilterWithIDs,     # ← MOTIP
    WrapInstanceIDs,             # ← MOTIP
)
```

**검증**: sample 0 (44 boxes) → filter 후 41 boxes / 41 IDs, 모든 sample에서 `box count == id count`

### Step 4: Head에 query_feat 노출 ✅

**파일**: `projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py`
**라인**: 631-652 (기존 outs dict 분기 두 곳 모두 수정)

**train mode (denoising padding이 있는 경우)** — 라인 637-645:
```python
# MOTIP: expose final-layer query feature, with the dn pad stripped
# so it indexes match all_cls_scores / all_bbox_preds.
query_feat = outs_dec[-1, :, mask_dict['pad_size']:, :]
outs = {
    'all_cls_scores': outputs_class,
    'all_bbox_preds': outputs_coord,
    'dn_mask_dict': mask_dict,
    'query_feat': query_feat,
}
```

**eval mode (dn_mask_dict=None)** — 라인 647-652:
```python
outs = {
    'all_cls_scores': all_cls_scores,
    'all_bbox_preds': all_bbox_preds,
    'dn_mask_dict': None,
    'query_feat': outs_dec[-1],  # MOTIP: [B, num_q+num_propagated, C]
}
```

**중요**: train 모드에서 outs_dec 앞부분에 denoising query가 padding으로 붙어있음. cls/bbox는 이미 잘라내고 있는데, query_feat도 같은 처리를 안 하면 인덱스 미스매치 발생.

- **검증**: forward 1 sample → `query_feat: (1, 428, 256)` 정확
- 동일 sample의 top-5 score는 변경 전후 동일 (read-only 노출 확인)

### Step 5: MOTIP 모듈 drop-in ✅

**새 폴더**: `projects/mmdet3d_plugin/models/motip/`

SparseBEV의 `models/motip/` 5개 파일을 복사 + 두 가지 cleanup:

| 파일 | 변경 | 내용 |
|------|------|------|
| `id_dictionary.py` | 그대로 | `IDDictionary` — K+1 learnable embeddings (K id + newborn) |
| `id_decoder.py` | tracker.py에서 rename | `IDDecoder` — `nn.TransformerDecoder` + `nn.Linear(d, K+1)` |
| `tracklet.py` | docstring 보강 | `TrackletFormer` — `concat([obj, pe, id])` → 3C token |
| `pos_encoding.py` | docstring 보강 | `Positional3DEncoding` — bbox+velo → C MLP (이름과 달리 sinusoidal 아님) |
| `__init__.py` | rewrite | 4개 클래스 re-export, `__all__` 명시 |
| `augmentation.py` | **삭제** | dead code (SparseBEV `compute_id_loss`가 inline 구현, 이 함수 호출 안 함) |

**검토했지만 그대로 둔 것**:
- `TrackletFormer.temporal_embed` — SparseBEV가 `rel_timestep`을 안 넘기므로 unused parameter. 향후 SparseBEV-trained 체크포인트 호환성 위해 유지. TODO 주석.

**검증** — 모든 모듈 instantiate + 1 forward pass 성공:
```
pe shape:        (10, 256)
spec shape:      (10, 256)
queries shape:   (1, 10, 768)
context shape:   (1, 30, 768)
logits shape:    (1, 10, 51)   ✓ (K=50 + 1 newborn)
```

### Step 6: Petr3D detector에 MOTIP loss 통합 ✅

**파일**: `projects/mmdet3d_plugin/models/detectors/petr3d.py`

**변경 내역**:

1. **import + `__init__`** (라인 17-19, 47, 64-83)
   - `IDDictionary`, `IDDecoder`, `Positional3DEncoding`, `TrackletFormer` import
   - `motip_cfg` 파라미터 추가
   - 4개 MOTIP submodule을 conditional로 instantiate (`motip_cfg=None`이면 기존 Petr3D와 동일)
   - `freeze_detector`, `id_loss_weight`, `context_len` flag 저장

2. **`forward_pts_train`** (라인 222-225)
   - head outs를 `self._motip_last_outs`에 cache (per-frame)

3. **`obtain_history_memory`** (라인 138, 149-152, 173-181)
   - `gt_instance_ids` 파라미터 추가
   - 매 frame loop마다 motip data (outs, gt, ego_pose, prev_exists)를 collect
   - clip 끝나면 `self._motip_frame_data`에 저장

4. **`forward`** (라인 273-282)
   - `gt_instance_ids` key가 있으면 transpose 처리에 추가

5. **`forward_train`** (라인 294, 348-353)
   - `gt_instance_ids` 파라미터 받기
   - obtain_history_memory 후 `compute_id_loss(self._motip_frame_data)` 호출
   - 결과를 losses dict에 update

6. **MOTIP helper 메서드 + `compute_id_loss`** (라인 363-555, 약 200줄, 새 메서드)
   - `_bbox_to_pe_input`: 10-dim StreamPETR bbox → 9-dim PE 입력
   - `_transform_bbox_to_current`: 좌표 변환 (이전 frame → 현재 frame lidar coord)
   - `compute_id_loss`: clip 단위 ID loss 계산
     - Step 1: 매 (b, t)마다 Hungarian matching → matched feat/bbox/raw_id
     - Step 2: clip 전체 ID re-indexing + random permutation augmentation
     - Step 3: 각 target frame t (1..T-1)에 대해 context (frames 0..t-1) 구성
       - 좌표 변환 + PE_3D + tracklet_former
       - Trajectory augmentation (occlusion + switch)
     - Step 4: ID prediction + cross-entropy loss
   - **결정 1-B 적용**: 전체 428 query 사용 (slicing 없음)
   - **SparseBEV 코드의 perf bug 두 가지 고침**: Python loop in torch.where → torch.isin 사용, GPU↔CPU sync 최소화

**검증**: queue_length=8 sliding window 학습 모드로 1 iteration:
```
loss_id:       4.0615  (finite ✓)   ← ≈ ln(51) random baseline
id_acc:        0.0209  (finite ✓)   ← ≈ 1/51 random
num_matched:   41.0000              ← Hungarian이 41개 GT 매칭
MOTIP grad:    115/116               ← temporal_embed (unused) 빼고 다 흐름
```

**주의**: `tracklet_former.temporal_embed.weight`는 unused (rel_timestep을 안 넘김) → DDP 학습 시 `find_unused_parameters=True` 필요.

### Step 7: Config 작성 ✅

**새 파일**: `projects/configs/StreamPETR/stream_petr_r50_motip_704_bs1_8key_24e.py`

base는 non-flash 60e config (architecture: 428q + NuImg pretrain). MOTIP을 위해 sliding window 학습 모드로 override:

| 항목 | base (60e seq) | 우리 (motip sliding) |
|------|----------------|----------------------|
| seq_mode | True | **False** |
| queue_length | 1 | **8** |
| num_frame_losses | 1 | **2** |
| num_frame_head_grads | 1 | **2** |
| num_frame_backbone_grads | 1 | **2** |
| seq_split_num | 2 | **1** |
| shuffler_sampler | InfiniteGroupEachSampleInBatchSampler | **DistributedGroupSampler** |
| samples_per_gpu | 2 | **1** |
| ann_file | *_train.pkl | **_train_motip.pkl** |
| pipeline | base | **WithIDs filters + WrapInstanceIDs + gt_instance_ids in keys** |
| optimizer.lr_mult | normal | **detector lr_mult=0 (frozen)** |
| load_from | None | **60e ckpt** |

motip_cfg: `freeze_detector=True` (Phase 1)

### Step 8: 1 iteration 검증 ✅

**새 파일**: `tools/verify_motip_train_step.py`

전체 파이프라인을 한 번에 검증:
1. dataset+pipeline build → gt_instance_ids 정상 흐름
2. model build + 60e ckpt load (MOTIP keys만 missing)
3. 1 batch forward → detection loss + id_loss 모두 finite
4. backward → gradient finite, MOTIP submodules에 흐름

결과는 위 Step 6 검증 결과 표 참조.

---

## 파일 변경 위치 요약

```
StreamPETR/
├── docs/MOTIP_INTEGRATION.md                  ← 이 문서
├── mmdetection3d                              ← symlink (셋업)
├── tools/prepare_motip_pkl.py                 ← Step 1 (새 파일)
├── tools/verify_motip_compat.py               ← 환경 검증용 (존재)
├── projects/configs/StreamPETR/
│   ├── stream_petr_r50_noflash_*.py           ← non-flash override (존재)
│   └── stream_petr_r50_motip_*.py             ← MOTIP config (Step 7)
├── projects/mmdet3d_plugin/
│   ├── models/motip/                          ← Step 5 (새 폴더)
│   ├── models/dense_heads/streampetr_head.py  ← Step 4 (1줄)
│   ├── models/detectors/petr3d.py             ← Step 6
│   ├── models/utils/attention.py              ← 셋업 (try/except, 완료)
│   ├── datasets/nuscenes_dataset.py           ← Step 2
│   ├── datasets/pipelines/transform_3d.py     ← Step 3
│   └── datasets/samplers/group_sampler.py     ← 셋업 (IPython 주석, 완료)
```

---

## Phase 1 학습 결과 (Sliding Window, iter_14064 = 4 epoch)

### Detection (mAP/NDS)

Detector frozen이므로 pretrained와 동일해야 함 → 확인됨.

```
mAP: 0.449   NDS: 0.546
```

### Training id_acc

```
iter   50: 18%
iter  500: 46%
iter 1000: 66%
iter 2500: 75%
iter 5000: 76%  ← plateau
iter 7000: 78%
```

78%에서 plateau. **빠르게 수렴 후 정체 → shortcut 학습 의심.**

### Tracking eval (AMOTA)

2-stage eval pipeline으로 측정 (tools/extract_track_feats.py → tools/eval_tracking.py):

```
AMOTA:    0.005    ← 사실상 tracking 안 됨
RECALL:   0.622    ← detection 자체는 잘 됨
MOTA:     0.015
IDS:      68,638   ← ID switch 매우 많음 (매 frame 새 ID 할당)
TP:       15,289
FP:       235,448
```

**결론: SparseBEV MOTIP_INTEGRATION_GUIDE.md 섹션 2.1과 동일한 현상.**

> Autoregressive buffer → wandb에서 85~92% id_acc가 나왔지만 eval에서 0%.
> 원인: buffer feature는 이전 step의 모델로 생성 → distribution mismatch.

StreamPETR의 sliding window (queue_length=8) 학습은 사실상 autoregressive buffer:
- 매 frame 순차 forward → buffer에 feature 저장 → 다음 frame에서 context로 재사용
- MOTIP weight 업데이트 후 이전 frame feature와의 distribution mismatch 발생
- 모델은 "대충 가장 비슷한 feature 매칭"이라는 shortcut 학습
- Inference에서는 이 shortcut이 동작하지 않아 ID가 매 frame 바뀜

### 원인 분석

| 현상 | 원인 |
|------|------|
| id_acc 78% plateau | persistent object의 trivial feature matching으로 달성 |
| IDS=68K | MOTIP decoder가 진짜 ID matching을 못 배움 |
| AMOTA=0.005 | shortcut 학습 → inference에서 transfer 실패 |
| Recall=0.622 | detection 자체는 정상 (detector frozen) |

---

## Step 9: 2-Stage Tracking Eval Pipeline ✅

### Stage 1: Feature 추출 (tools/extract_track_feats.py)

`dist_test`와 동일 경로로 detection + query_feat 추출. DataLoader 사용으로 빠름.

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=... PYTHONPATH=... \
python tools/test.py <config> <ckpt> --launcher pytorch --out <output.pkl>
```

- petr3d.py의 `simple_test_pts`에서 `query_feat`, `bbox_raw`, `cls_scores` 함께 저장
- 6019 samples, ~7분, ~4.2GB pkl

### Stage 2: Tracking + Eval (tools/eval_tracking.py)

pkl을 읽어서 MOTIPTracker 실행 → nuScenes submission JSON 생성 → TrackingEval.

```bash
python tools/eval_tracking.py \
  --config <config> --checkpoint <ckpt> --feats <pkl> \
  --det-thresh 0.05 --new-thresh 0.1 --id-thresh 0.1
```

- Tracker: ~10분 (150 scenes)
- Eval: ~4분
- 좌표 변환: SparseBEV val_tracking.py의 `lidar_to_global` 이식 (gravity center 직접 변환)

### Visualization (tools/viz_tracking_cam.py)

6개 카메라 뷰에 3D tracking box를 projection한 mp4 생성.

```bash
python tools/viz_tracking_cam.py \
  --submission <tracking_results.json> --scene scene-0003
```

### 주요 디버깅 기록

1. **data_infos 순서 mismatch**: `val.pkl`의 infos 순서와 `dataset.data_infos` 순서가 다름. DataLoader는 dataset 순서를 따르므로, pkl의 infos가 아닌 `dataset.data_infos`를 사용해야 함.
2. **DETR 계열 detection score**: StreamPETR은 score>0.3인 detection이 frame당 2~3개뿐. `det_thresh=0.05`로 낮춰야 함.
3. **gravity center vs bottom center**: `denormalize_bbox`는 gravity center 반환, `LiDARInstance3DBoxes`는 bottom center 기대. 직접 변환 시 주의.

---

## Step 10: Scene Boundary Bug 수정 ✅

### 버그 발견

Phase 1 학습 결과 분석 중 **AMOTA=0.005, IDS=68K**로 tracking이 사실상 안 되는 문제 확인.

Detector frozen이므로 distribution mismatch 가설은 해당 없음. 실제 원인:

**`compute_id_loss`가 scene 경계를 무시하고 다른 scene의 frame을 context에 포함.**

StreamPETR의 sliding window (queue_length=8)는 dataset 순서대로 연속 8 frame을 가져오는데, scene 경계를 넘을 수 있음:

```
Queue: [SceneA_38, SceneA_39, SceneA_40 | SceneB_1, SceneB_2, SceneB_3, SceneB_4, SceneB_5]
                                         ↑ prev_exists=0 (scene 전환)
```

실측: **training sample의 17.4% (4893/28123)에서 scene 경계가 queue에 포함.**

이 경우 `compute_id_loss`가 SceneA의 object ID와 SceneB의 object ID를 같은 clip의 ID re-indexing에 혼합 → garbage label로 학습.

### 수정 내용

**파일**: `projects/mmdet3d_plugin/models/detectors/petr3d.py`, `compute_id_loss`

`prev_exists`를 사용해서 각 batch의 scene 시작점을 찾고, 해당 scene 내의 frame만 사용:

```python
# scene boundary 탐지
scene_start = [0] * B
for b in range(B):
    for t in range(T):
        if prev_exists[t][b] == 0:
            scene_start[b] = t

# ID re-indexing: scene_start 이후 frame만
scene_frames = [b_matched[t] for t in range(ss, T) if b_matched[t] is not None]

# context 구성: scene_start 이후 frame만
for t in range(ss + 1, T):
    for c in range(ss, t):  # ← 이전: range(0, t)
```

### SparseBEV에서 이 문제가 없었던 이유

SparseBEV는 `DistClipSampler`로 **항상 같은 scene 내의 연속 frame만 샘플링**. Scene 경계가 clip에 절대 포함되지 않음.

---

## 다음 단계

Phase 1을 scene boundary fix 적용 후 재학습하여 tracking 성능 확인.

---

## 미해결 / 후속 작업

1. ~~**Full eval로 mAP/NDS 재현 검증**~~ → ✅ mAP=0.449, NDS=0.546 확인
2. **Non-autoregressive clip training 구현** — Phase 1 shortcut 문제 해결을 위해 필수
3. **Newborn/existing accuracy 분리 로깅** — shortcut 학습의 정확한 패턴 파악
4. **Switch augmentation 검증** — context ID만 swap하고 query GT는 그대로 두는 현재 구현이 MOTIP 원본과 일치하는지 확인
5. **flash-attn 설치** — 학습 속도/메모리 개선
6. **CPU contention** — 다른 사용자의 학습 프로세스로 시스템 load average 50+. GPU 0,1만 사용 + workers_per_gpu 최소화
