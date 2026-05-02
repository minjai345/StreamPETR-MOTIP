# MOTIP 학습·실험 진행 흐름 정리

- 작성일: 2026-04-21
- 기간: 2026-04-20 ~ 2026-04-21
- 모노트: [[MOTIP_논문노트_20260420]]
- 관련: [[MOTIP_심화_ICL과ID임베딩_20260420]], [[MOTIP_코드매핑_20260420]], [[MOTIP_비판적분석_20260420]]
- 목적: 논문 이해 → 비판적 분석 → StreamPETR+MOTIP 실험 진단으로 이어진 2일간의 흐름을 시간순·가설변화순으로 정리. 앞으로 되돌아와 맥락 복구할 때 이 문서만 보면 충분하도록.

---

## 🗺️ 한눈에 보는 타임라인

```
[Phase 1 이해] → [Phase 2 심화] → [Phase 3 코드검증] → [Phase 4 비판]
    ↓
[Phase 5 inference AR 구조 검증]
    ↓
[Phase 6~8 실험 진단 — 가설 3회 정정]
    ↓
[Phase 9 용어 명확화] → 현재
```

---

## Phase 1 — 논문 기초 이해

**핵심 파악**
- MOTIP = "MOT를 in-context ID classification으로 재정의"
- Deformable DETR detection + ID Decoder association **분리 구조**
- MOTR 계열(tracking-by-query)과 달리 detection/association supervision conflict 없음

**읽기 전 3대 질문**
1. "in-context ID prediction"이 구체적으로 뭔가? (Transformer Q/K/V 관점)
2. Learnable ID Dictionary가 어떻게 permute-invariant하게 되나?
3. MOTR 계열과 본질적 차이는?

**산출물**: [[MOTIP_논문노트_20260420]]

---

## Phase 2 — 심화 개념 Q&A

**혼란 지점들**
- "in-context ID prediction이 이해 안 간다"
- "NLP ICL 비유 부분을 모르겠다"
- "왜 learnable embedding? PE 안 쓰고?"

**해결한 방법**
- NLP few-shot ICL (GPT-3 스타일)과 MOTIP 구조 1:1 매핑
- ID embedding 학습 역학을 **Neural Collapse / ETF** 관점으로 설명
- PE 대신 learnable을 쓰는 이유: ID는 **semantic-free slot**이라 positional 구조 필요 없음

**산출물**: [[MOTIP_심화_ICL과ID임베딩_20260420]] (8개 Q&A 섹션)

---

## Phase 3 — 코드 매핑 검증

**검증하고 싶었던 직관**
> "매 training iteration마다 trajectory에 붙는 ID가 랜덤 permute됨 = 같은 scene을 또 학습해도 ID는 바뀐다"

**코드로 확인**: `transforms.py:425`
```python
_random_id_labels = torch.randperm(self.num_id_vocabulary)[:_N]
_random_id_labels = _random_id_labels[None, ...].repeat(_T, 1)
id_labels[group] = _random_id_labels.clone()
```
→ 직관 **정확** 확인

**더 확인한 것들**
- `id_decoder.py` Line 108-112: trajectory token concat $\tau = [f; i^k]$
- Line 119: causal time mask (`trajectory_times >= unknown_times`)
- Line 171: classifier는 decoder output의 `[..., -id_dim:]`만 씀 (구조적 비대칭)

**산출물**: [[MOTIP_코드매핑_20260420]] (논문 §3 Method ↔ 코드 1:1)

---

## Phase 4 — 비판적 분석

**논문 §5 Limitations (저자 인정)**
- $K=50$ 용량, token 활용률 <40%
- Tailored ID Decoder 부재
- 추가 cues (motion/depth/appearance) 부재
- 정교한 trajectory 모델링 미적용

**저자가 놓친 blind spot (발굴)**
1. ⭐ **Exposure bias** — training(TF) vs inference(AR) 분포 불일치
2. **Sequence-level supervision 부재** — frame-wise CE만
3. **Detection-Association 분리의 cascading error**
4. "In-context" 용어의 수사적 사용

**Autoregressive 확장 3가지 레시피**
- Full AR Rollout (❌ 현실적 불가)
- Scheduled Sampling (⭐⭐⭐ 추천)
- Sliding Window + Detach (⭐⭐)

**산출물**: [[MOTIP_비판적분석_20260420]]

---

## Phase 5 — Inference 구조 검증

**질문**: "inference는 autoregressive가 맞나?"

**코드 추적** (`runtime_tracker.py`)
```
detect → _get_id_pred_labels(context=self.trajectory_*) → 예측 ID
       ↓
_update_trajectory_infos: 예측 ID를 self.trajectory_id_labels에 append
       ↓
다음 frame에서 이 업데이트된 buffer가 context로 재사용
```

**결론**: YES, inference는 autoregressive. Training(TF) vs Inference(AR) 분포 불일치가 **코드 레벨에서 실증됨** = exposure bias의 존재 증거.

---

## 🧪 실험 진단 — 가설 3회 수정 이력

> **StreamPETR + MOTIP nuScenes val 실험**. AMOTA 양호(~0.39), IDS ~2800으로 과다.

### Round 0: Exposure Bias 주범 의심

- Claude 첫 가설: "teacher forcing만으로 학습 → inference 시 context 오염으로 drift"
- 추천 처방: Scheduled Sampling 도입
- 진단 제안: GT-feeding 실험

### Round 1: Under-Detection 주범 (첫 정정)

**증거**: 150개 val scene per-frame IDS 시각화
- 파랑(#GT), 녹색(#Pred), 빨강(IDS) overlay
- **관찰**: Pred < GT 간극 클 때 IDS 집중 / Pred≈GT scene은 IDS≈0
- 최악 scene (0105, 0276, 0018, 0101, 0916)은 모두 dense + under-detection

**정정된 가설**
```
Detection FN (under-detection)
    ↓
Track continuation 실패 → miss_tolerance 종료
    ↓
재등장 시 newborn 판정 → IDS
```

**새 처방**: det_thresh↓, miss_tolerance↑, warm-up 처리

### Round 2: Association 자체가 Bottleneck (두 번째 정정)

**증거**: 12개 threshold 조합 sweep

| # | det | new | id | age | AMOTA | IDS |
|---|---|---|---|---|---|---|
| **2** | **0.25** | **0.40** | **0.10** | **10** | **0.3919** | **2727** |
| 3 | 0.25 | 0.40 | 0.10 | 5 | 0.3918 | **2547** (최저) |
| 1 | 0.20 | 0.40 | 0.10 | 10 | 0.3922 (최고) | 3262 |
| 6 | 0.15 | 0.40 | 0.10 | 5 | 0.3765 | 3871 |
| 7~12 | — | **0.30** | — | — | 0.37 내외 | **6500+** |

**결정적 발견**
1. `det_thresh↓` (0.25→0.20→0.15) 하면 오히려 **IDS 증가** — FP noise가 주범
2. `newborn_thresh=0.30`은 재앙 (IDS 2.5배)
3. **`max_age` 변화 거의 무관** ← 결정적 단서

**`max_age` 무관이 말해주는 것**
- Track 끊겼다가 다시 복구하는 miss-and-recovery 시나리오가 주 원인이 **아님**
- 대신 **살아있는 track 사이에서 ID swap**이 일어남을 강력히 시사

**새 가설 (현재)**
**MOTIP의 ID Decoder가 3D 환경에서 association을 충분히 discriminative하게 못 함**

의심 메커니즘:
- (a) **Motion cue 부재** — trajectory token이 $(f, i^k)$만, 3D 위치·속도 명시 없음
- (b) ID Dictionary entanglement
- (c) Exposure bias는 보조 증폭 요인 (주동력 아님)

### 가설 변화 요약표

| 단계 | 주범 가설 | 증거로 반증된 방식 |
|---|---|---|
| Round 0 | Exposure bias | 다음 단계에서 정정 |
| Round 1 | Under-detection (detection FN) | Threshold sweep에서 det_thresh↓가 오히려 악화 |
| Round 2 | Association 자체(motion cue 부재) | 현재 — GT-feeding으로 검증 대기 |

---

## Phase 9 — 용어 명확화

### Exposure Bias
**정의**: 학습(GT context) vs 추론(predicted context) 분포 불일치로 모델이 **자기 예측이 섞인 입력**을 학습에서 본 적 없어 inference 시 drift 발생.

**MOTIP에서의 모습**
- Training context: $[\text{GT}_{t-3}, \text{GT}_{t-2}, \text{GT}_{t-1}]$
- Inference context: $[\text{pred}_{t-3}, \text{pred}_{t-2}, \text{pred}_{t-1}]$

한 프레임 ID 오류가 cascade되는 구조적 취약점.

### GT-Feeding 실험
**목적**: "만약 inference 때도 GT context를 강제 주입했다면 IDS가 얼마나 나오는가?" → exposure bias 기여도를 isolate.

**구현**: `runtime_tracker.py`의 `_update_trajectory_infos`에서 `id_labels`를 GT로 교체 (5줄 패치).

**진단 룰**
| 결과 | 해석 |
|---|---|
| IDS 대폭 감소 (2727 → ~500) | Exposure bias 주범 |
| IDS 비슷 (2727 → 2500) | Association 자체 약함 |
| IDS 중간 감소 | 둘 다 기여 |

---

## 🧭 현재 위치 & Open Questions

### ✅ 확정된 것
- Threshold tuning의 AMOTA 상한 ≈ 0.39 (구조적 한계)
- IDS 문제의 성격 = "살아있는 track 사이 ID swap", "track 끊김 복구" 아님
- 순수 hyperparameter 튜닝으로는 IDS 2500 벽 못 뚫음

### ❓ 결정적 실험 대기 중
1. **GT-feeding** — Exposure bias 기여도 decisive 측정
2. **Motion cue 주입 ablation** — Trajectory token에 3D relative position concat
3. **IDS 유형 분해** — same-class swap vs cross-class vs newborn 비율

### Best Configuration (현재)
```
det_thresh=0.25, newborn_thresh=0.40, id_thresh=0.10, max_age=10
→ AMOTA 0.3919, IDS 2727
```

---

## 📦 생성된 산출물 (전체 목록)

| 파일 | 역할 |
|---|---|
| [[MOTIP_논문노트_20260420]] | 마스터 노트 (§1~6 구조) |
| [[MOTIP_심화_ICL과ID임베딩_20260420]] | ICL/embedding 심화 Q&A |
| [[MOTIP_코드매핑_20260420]] | 논문 §3 ↔ code 1:1 |
| [[MOTIP_비판적분석_20260420]] | §5 + blind spot + 확장 아이디어 |
| **[[MOTIP_진행흐름_20260421]]** | 본 문서 (전체 흐름 정리) |
| `memory/project_streampetr_motip_ids.md` | 실험 상태 + 가설 변화 이력 (persistent) |

---

## 🎯 한 줄로

> **MOTIP 구조 완전 이해 → 논문이 안 다룬 blind spot(exposure bias)을 발굴 → 3D(StreamPETR)로 옮겨 실험하니 IDS 2700대 → 가설을 두 번 수정하며 진짜 원인이 "association 자체의 discriminability 부족(motion cue 부재)"임을 좁혀온 상태. 다음은 GT-feeding으로 exposure bias 기여도를 분리 확증.**

---

## 🔜 다음 액션

- [ ] GT-feeding v2 결과 검토 (실행 중) — pool exhaustion 해소 후 IDS 측정
- [ ] Motion cue 주입 ablation — trajectory token에 relative $(\Delta x, \Delta y, \Delta z)$ concat
- [ ] IDS 유형 분해 — nuScenes eval script에서 same-class/cross-class/newborn 비율
- [ ] 결과가 들어오면 [[MOTIP_비판적분석_20260420]] §2-4 (Detection-Association 분리의 양면성) + §4-3 (3D temporal cue) 강화/수정

---

## Phase 10 — GT-feeding 구현 시도와 재설계 (2026-04-22)

### 도구 위치
모든 oracle/diagnostic 스크립트는 `tools/experiments/` (uncommitted):
- `oracle_association_eval.py`: matched det의 출력 tracking_id만 GT로 swap (decoder/buffer 무손상). 결과 AMOTA=0.4469, IDS=256 — "matched 외 IDS 하한".
- `gt_feeding_eval.py`: id_emb 만 GT slot 오버라이드, slot_to_global은 그대로. AMOTA=0.3284, IDS=6643 (악화) — slot space 해석 불일치로 실패.
- `gt_aligned_eval.py`: id_emb + slot_to_global 둘 다 GT 공간으로 align (`GTAlignedTracker`).

### 실행 환경 노트
- env: `conda activate sparsebev` + `LD_LIBRARY_PATH=/data4/minjae/miniforge3/envs/sparsebev/lib:$LD_LIBRARY_PATH` 필수 (mmcv ext_module 로딩 시 GLIBCXX 3.4.29 요구)
- checkpoint: `work_dirs/motip_e2e_v2_fresh/iter_28128.pth`
- pre-extracted feats: `work_dirs/motip_e2e_v2_fresh/track_feats_iter28128.pkl` (Stage 1 산출, 4.4GB)
- tracker만 ~2분, nuScenes TrackingEval은 ~35분 (40 recall threshold × 7 class × py-motmetrics single-thread)

### v1 시도 (실패) — Strict newborn 해석 오류

**스펙 5단계** ("Unmatched detection (FP)은 기존 newborn 로직 유지")를 **"matched + decoder-newborn → 새 slot 할당"**으로 잘못 풀어 구현.

결과 (2026-04-22 00:08~00:39):
- AMOTA=**0.3757** (baseline 0.3918 → **악화**)
- IDS=**4421** (baseline 2547 → **+1874 악화**)
- Per-outcome counters:
  - matched_correct=48437 / matched_wrong_slot=5191 / matched_newborn=2164 / matched_dropped=7477
  - **matched_pool_full=15725** ← K=50 슬롯 풀 고갈

**원인 — Ghost slot leak:**
1. `_register_gt_instance`가 GT instance 첫 매칭 시 `slot_to_global[s]=gid_I` 영구 등록
2. Decoder가 그 slot을 안 고르면 (wrong/newborn/dropped) `gid_I`는 `self.tracks`에 미진입
3. age 평가 루프 `for gid in self.tracks` → `gid_I` 빠짐 → evict 안 됨
4. → slot s가 `slot_to_global`에 영구 잔존 (ghost slot)
5. Strict-newborn까지 추가하면 matched-newborn 이벤트당 새 slot도 할당 → K 고갈 가속

K=50 시나리오에서 약 20% matched 이벤트 (15725/78994)가 pool_full로 unmatched path로 떨어져 매 frame 새 gid → IDS 폭증. 측정 자체가 오염.

저장 위치: `work_dirs/motip_e2e_v2_fresh/gt_aligned_strict_v1_buggy/`

### v2 재설계 결과 (2026-04-22 12:41)

**핵심 수정 3가지 (`gt_aligned_eval.py` patch):**

1. **Matched + decoder-newborn**: slot 할당 폐기. fresh `tracking_id`만 출력 (`slot_to_global` 미수정). nuScenes eval에서 IDS로 카운트되되 slot pool 잠식 없음.
2. **GT 관측 시 self.tracks 강제 갱신**: matched det의 모든 outcome (correct/wrong/newborn/dropped)에 대해 `self.tracks[gid_I].age=0`. 관측 중인 GT slot은 절대 evict 안 됨.
3. **`_free_slot` 시 inst_to_gid 동시 청소**: gid_to_inst reverse map 추가, 슬롯 free 시 inst_tok 매핑도 제거 → stale registration 차단.

저장 위치: `work_dirs/motip_e2e_v2_fresh/gt_aligned_v2/`

### v2 결과 수치

- AMOTA=**0.3791** (baseline 0.3918, v1 0.3757)
- MOTA=0.3501
- IDS=**4048** (baseline 2547, v1 4421) ← 여전히 baseline보다 높음

**Per-outcome counters:**
- matched_correct=49566 / wrong_slot=5294 / newborn=2090 / dropped=7864
- matched_pool_full=**14180** (v1 15725 → -1545, ghost-leak 일부 회복했지만 K=50 자체 병목 남음)
- structural decisions: 7384 / 64814 matched = **11.4%**

**Per-scene 분석 (K=50 도달 여부로 분리):**

| Subset | Scenes | Matched | Correct | Structural |
|---|---|---|---|---|
| Clean (pool_full=0) | 91 | 34026 | **78.8%** | **9.4%** |
| Polluted (pool_full>0) | 59 | 30788 | 73.9% | 13.6% |

→ Clean scene 91개에서 **decoder structural failure rate = 9.4%**. 이게 가장 깨끗한 신호.

### 해석 (잠정)

**Decision-rate 기반 추정:**
- Clean 조건에서 decoder는 matched det의 **78.8% 정답**, **9.4% 구조적 실패**, ~11.8% 미확신(dropped).
- 만약 decision-rate가 IDS-rate에 선형 대응한다고 가정하면:
  - Baseline의 matched-IDS 2291건 중 ~9% (≈215) = structural 기여
  - 나머지 ~91% (≈2076) = exposure bias 기여
- **이 결과는 Round 2의 "association이 주범" 가설과 충돌**. Clean context 주면 decoder가 대부분 맞춤.

**경고:** decision-rate ≠ IDS-rate (event-level 변환 비선형). Strong claim 위해서는 baseline / oracle / v2를 동일한 91 clean scene subset에서 nuScenes eval 재실행 필요.

### K=50 병목

- 150 scene 중 **39% (59개)**가 pool_full hit. max=1444 events/scene.
- v2 redesign으로 ghost는 막았지만, scene 내 active GT instance 수가 K를 넘는 경우는 본질적 한계.
- MOTIP의 `K=50` 학습 hyperparameter 자체가 nuScenes urban 밀집 scene에 부족할 수 있음.

### 다음 단계 후보 (v2 결과 직후 시점에서)

- (P1) 91 clean scene subset에서 baseline / oracle / v2 셋 다 nuScenes eval 재실행 → IDS 직접 비교
- (P2) K=100/200으로 모델 retrain → pool 압박 제거하고 정확한 attribution
- (P3) scene별 IDS 분포 분석 (busy vs quiet) → 가설 정교화

---

## Phase 11 — Clean-91 Subset 비교 (P1 실행, 2026-04-22 14:27)

### 도구
`tools/experiments/eval_clean_subset.py`: `gt_aligned_counts.json`의 per_scene `matched_pool_full=0`인 91개 scene만 골라, 기존 tracking_results JSON을 filter 후 `nuscenes.utils.splits.create_splits_scenes` 패치하여 TrackingEval 재실행. Smoke test (3 scene) 통과 후 본 실행.

### 결과

| | Baseline | Oracle | v2 |
|---|---|---|---|
| AMOTA (Full 150) | 0.3918 | 0.4469 | 0.3791 |
| IDS (Full 150)   | 2547   | 256    | 4048 |
| **AMOTA (Clean 91)** | **0.4192** | **0.4527** | **0.4010** |
| **MOTA (Clean 91)**  | 0.3985 | 0.4313 | 0.3887 |
| **IDS (Clean 91)**   | **154** | **63** | **909** |

**Clean 91 matched-side IDS** (= total − oracle):
- Baseline: 91
- v2 (raw): 846 (≈ 9.3× baseline)

**v2 clean-91 per-decision counts** (matched_total=34026):
- matched_correct: 26823 (78.8%)
- matched_wrong_slot: 2197 (6.5%)
- matched_newborn: 986 (2.9%)
- matched_dropped: 4020 (11.8%)

### 측정 비대칭 발견

**v2가 baseline보다 IDS가 많아진 결정적 이유 = matched_newborn 처리 비대칭.** GT inst가 5프레임 관측되는데 중간 1프레임만 decoder가 newborn 판정한 시나리오:

```
baseline:
  frame t  : decoder newborn → 새 slot s' 할당, slot_to_global[s']=gid_new, emit gid_new
  frame t+1: decoder가 history에 gid_new id_emb 보고 slot s' 재선택 → emit gid_new (자기 일관)
  → nuScenes IDS: gid_old → gid_new = 1 IDS

v2:
  frame t  : matched-newborn → emit fresh gid_99 (slot_to_global 미수정, slot은 여전히 gt_gid 소유)
  frame t+1: decoder가 gt_slot 선택 → emit gt_gid (history GT-clean이라 원래 slot 복원)
  → nuScenes IDS: gt_gid → gid_99 → gt_gid = 2 IDS
```

→ v2 newborn 1건당 IDS 2개, baseline 1개. 986건 × 2 ≈ 1972 IDS 상한. 실측 909 중 상당 부분이 이 artifact.

**비대칭 보정 후 추정**: v2 matched-IDS ≈ 400~500 (baseline 91의 4-5배).

### 해석 (수정됨)

이전 추정 ("exposure bias가 91% 기여")은 **측정 artifact 영향이 컸음**. 더 정확한 해석:

1. **Decoder는 self-consistency가 강함**: baseline에서 95%+ 자기 일관성으로 IDS=91만 발생. Autoregressive context가 self-reinforcing하게 작동 — 한번 slot 할당하면 history feedback으로 같은 slot picking 유지.

2. **하지만 GT-correctness는 78.8%로 낮음**: v2에서 history를 GT-clean으로 강제하면, decoder는 매 frame 처음부터 inst→GT slot 매칭을 해야 하는데 이게 어려움. 21.2% 비율로 wrong/newborn/dropped.

3. **두 metric의 격차 = decoder가 "절대 정체성"이 아닌 "context-relative tracking"을 학습했다는 증거**: 학습 시 random slot permutation 때문에 decoder는 "history와 일관되게 picking"만 학습. 외부에서 GT-aligned slot을 강제하면 그 능력이 노출됨.

4. **"Exposure bias vs structural" framing은 부정확**: 
   - Exposure bias는 baseline에서 helpful (consistency loop) — 이전 가정과 반대
   - Structural weakness는 진짜 존재 (78.8% GT-correctness)
   - 하지만 baseline 환경에서는 self-reinforcement로 가려짐

### 따라서 처방 우선순위 재고

이전: 가설 = "association 자체 약함" → 처방 = motion cue 추가 (decoder 능력 강화)

수정안:
- (Q1) **Scheduled Sampling 효과 재평가**: exposure bias가 harmful drift가 아닌 helpful loop라면 SS는 효과 미미할 수 있음. 단 학습 분포 정렬 측면에선 여전히 유효 가능.
- (Q2) **Motion cue 주입 ablation은 여전히 유효**: 78.8% GT-correctness 자체를 끌어올리는 직접적 처방. 현 가설에서 가장 결정적 실험.
- (Q3) **K=50 capacity 자체가 nuScenes urban scene에 부족**: AMOTA 상한 0.39 중 일부는 capacity bottleneck 기여. K↑ retrain의 가치 재평가 필요.

### 데이터 위치
- v2 (full): `work_dirs/motip_e2e_v2_fresh/gt_aligned_v2/`
- v1 buggy (보존): `work_dirs/motip_e2e_v2_fresh/gt_aligned_strict_v1_buggy/`
- Clean-91 subset evals: `work_dirs/motip_e2e_v2_fresh/{,work_dirs/motip_e2e_v2_fresh/}clean91_{baseline,oracle,v2}/`

---

## Phase 12 — IDS Event 분해 + Slot Flip + Proximity 진단 (2026-04-22 ~ 05-01)

Round 5 가설을 결정적으로 검증하고 처방 우선순위를 데이터로 잠그기 위해 3개 진단을 진행.

### 12-A. IDS Event 분해 (실험 A)
**도구:** `tools/experiments/ids_event_decompose.py`

**방법:** Per-class 2m Hungarian matching (single operating threshold). Precedence-based 분류 — track_break_recover (gap≥2) → newborn_fp (new pred_id 처음 등장) → same_class_swap. Cross-class swap은 nuScenes per-class eval 구조상 정의 불가, secondary diagnostic으로 분리. Detection_fail (class-internal match rate <50%)도 별도 카운트.

**Methodology caveat:** 우리 single-threshold count = 4502 ≠ nuScenes IDS=2547 (40 recall threshold AMOTA-style 평균). 비율(%) 신뢰, 절대수는 approximate.

**Results (baseline iter_28128, full 150):**

| Category | Events | % |
|---|---|---|
| same_class_swap | 2320 | **51.5%** |
| newborn_fp | 1919 | 42.6% |
| track_break_recover | 263 | 5.8% |

**Per-class 시그니처:**

| Class | track_break | newborn_fp | same_class | total | dominant |
|---|---|---|---|---|---|
| pedestrian | 158 | 820 | **1520** | 2498 | same-class (61%) |
| car | 81 | **861** | 669 | 1611 | newborn (53%) |
| truck | 7 | **113** | 48 | 168 | newborn (67%) |
| bicycle | 4 | 40 | 50 | 94 | mixed |
| motorcycle | 8 | 36 | 10 | 54 | newborn (67%) |
| bus | 1 | 35 | 5 | 41 | newborn (85%) |
| trailer | 4 | 14 | 18 | 36 | mixed |

→ **Pedestrian이 same-class swap의 65.5%** (1520/2320). Vehicles는 newborn_fp 지배.

**Secondary:**
- detection_fail: 4388 trajectory (총 7829 = **56%**). AMOTA 0.39 상한의 주범.
- cross_class_misdetection: 230 (5%만). detector class confusion bottleneck 아님.

저장: `work_dirs/motip_e2e_v2_fresh/ids_decompose_baseline/`

### 12-B. Slot Flip Rate (실험 B)
**도구:** `tools/experiments/slot_flip_rate.py`

**목적:** Round 5 직접 검증 — baseline의 self-consistency rate vs v2의 GT-correctness rate 비교.

**3 metric 정의:**
- adj_strict: P(pid_t == pid_{t-1} | matched at both t and t-1)
- persistent: P(pid_t == pid_first | matched at t, t > first)
- with_drop: P(pid_t == pid_{t-1} | matched at t-1)

**Results (clean 91 scenes, fair subset):**

| Metric | Baseline | v2 | Gap |
|---|---|---|---|
| adj_strict | 97.0% | 91.5% | +5.5pp |
| **persistent** | **82.9%** | **47.6%** | **+35.3pp** ★ |
| with_drop | 92.9% | 87.6% | +5.3pp |

**Round 5 결정 분기 결과:** persistent gap +35.3pp ≫ 15pp threshold → **Round 5 강하게 확증**. baseline은 lifetime ID stability를 autoregressive context의 self-reinforcing loop으로 유지, v2는 GT 강제 정렬 시 절반만 유지.

**Per-class persistent gap (clean91):**
| Class | Baseline | v2 | Gap |
|---|---|---|---|
| trailer | 98.1% | 37.5% | +60.6pp |
| bicycle | 89.3% | 24.0% | +65.3pp |
| motorcycle | 84.7% | 32.7% | +52.0pp |
| car | 88.5% | 50.7% | +37.8pp |
| truck | 78.4% | 51.1% | +27.3pp |
| bus | 91.5% | 65.0% | +26.5pp |
| pedestrian | 60.9% | 35.4% | +25.5pp |

→ Vehicles에서 gap 매우 큼. Pedestrian은 둘 다 낮지만 gap은 유사.

**12-B 추가 검증:** class-agnostic matching (v2 INTERNAL과 동일 잣대) 재측정 → class-aware vs class-agnostic 차이 0.2pp 이내. 매칭 알고리즘 차이는 결과에 영향 없음. v2 INTERNAL `matched_correct=78.8%` vs slot_flip 47.6% 차이는 metric 정의 자체가 다름 (decision-level vs JSON-based stability) — apples-to-oranges, 그러나 slot_flip 자체 비교는 fair.

저장: `work_dirs/motip_e2e_v2_fresh/slot_flip_{baseline,v2,baseline_agn,v2_agn}/`

### 12-C. Pedestrian same_class_swap Proximity vs Feature 진단 (2026-05-01)
**도구:** `tools/experiments/swap_proximity_analysis.py`

**목적:** Pedestrian same_class_swap 1520건의 root cause가 spatial proximity인지 visual feature similarity인지 결정. Identity-aware contrastive loss vs motion cue 처방 효과 추정.

**방법:** 각 swap event (P가 Y로 swap)에 대해:
1. P' = "Y의 원래 owner GT" 식별 (matched_per_scene에서)
2. P와 P'가 모두 매칭됐던 common_frame에서:
   - 3D distance(P, P')
   - cosine_similarity(P의 매칭 pred query_feat, P'의 매칭 pred query_feat) — `track_feats_iter28128.pkl` 활용

**Results (1214/1520 분석, no_owner 300건 분석 불가):**

| Quadrant | 개수 | % | 처방 |
|---|---|---|---|
| **NEAR + SIMILAR** (≤3m, cos≥0.9) | 942 | **77.6%** | motion OR contrastive 모두 효과 |
| NEAR + DISSIM | 142 | 11.7% | proximity 단독 → motion cue로 풀림 |
| FAR + SIMILAR | 34 | **2.8%** | feature 단독 → contrastive로 풀림 |
| FAR + DISSIM | 96 | 7.9% | 다른 원인 (context 오염) |

**Marginal 통계:**
- Distance: median 1.07m, **89.3%가 ≤3m**
- Cos sim: median 0.965, **80.4%가 ≥0.90**
- 두 신호 strongly correlated — feature similarity는 proximity의 부산물

**Single-prescription 효과 추정:**

| 처방 | 단독 적용 시 풀릴 케이스 |
|---|---|
| Motion cue | NEAR 전체 (1084) = **89.3%** |
| Contrastive | SIMILAR 전체 (976) = 80.4% |
| Motion만 (NEAR + DISSIM) | 142 = 11.7% |
| Contrastive만 (FAR + SIMILAR) | 34 = **2.8%** |

→ **Motion cue가 거의 sufficient.** Contrastive 추가시 +2.8% 한계 효과.

저장: `work_dirs/motip_e2e_v2_fresh/ids_decompose_baseline/swap_diag_pedestrian.jsonl`

### 12-D. 처방 우선순위 (Data-locked)

| Priority | 처방 | Target IDS | 근거 |
|---|---|---|---|
| **1순위** | Motion cue 주입 retrain | same_class 51.5% × 89.3% (proximity 분리) + vehicles의 일부 newborn_fp | Phase 11 Round 5 + Phase 12-C |
| 2순위 | Scheduled Sampling 또는 longer context | newborn_fp 42.6% (특히 vehicles 53-85%) | 12-A per-class signature |
| 3순위 | Detection 측 개선 (backbone, multi-scale) | detection_fail 4388 trajectory (56%) → AMOTA 상한 끌어올림 | 12-A secondary |
| 4순위 | Identity-aware contrastive loss | +2.8% marginal (FAR+SIMILAR) | 12-C |
| 낮은 | ReID / max_age | track_break 5.8% | 12-A |
| 낮은 | K=50 capacity↑ retrain | 39% pool_full scene 영향 (이미 상당 부분 v2 redesign으로 대응) | Phase 11 |

### 12-E. 산출물 위치 (전체)

**스크립트:** `tools/experiments/`
- `gt_aligned_eval.py` — GT-feeding tracker (v2)
- `eval_clean_subset.py` — Clean-91 subset eval wrapper
- `ids_event_decompose.py` — IDS event 분류 (실험 A)
- `slot_flip_rate.py` — Slot flip rate 측정 (실험 B)
- `swap_proximity_analysis.py` — Proximity vs feature 진단 (실험 C)

**결과:** `work_dirs/motip_e2e_v2_fresh/`
- `gt_aligned_v2/` — GT-feeding v2 결과 + counts
- `gt_aligned_strict_v1_buggy/` — v1 (ghost-leak) 보존
- `clean91_{baseline,oracle,v2}/` — subset eval
- `ids_decompose_baseline/` — IDS event jsonl + summary + swap_diag_pedestrian.jsonl
- `slot_flip_{baseline,v2,baseline_agn,v2_agn}/` — consistency_summary.json
