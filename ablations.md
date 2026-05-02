# Ablations

Branch: `5x4_from_scratch`. Each ablation is a full start-to-end training run from a freshly-initialized model. Only the listed flags differ from the baseline; everything else matches the baseline.

## Ablation 1 — Baseline

The configuration used in `results/sgd_gate_run/`.

**Optimizer**
- SGD, momentum 0.9
- lr=1e-2, lr-milestones=[115, 170] (×0.1 drop)
- weight_decay=1e-4

**Training**
- 600 iterations
- 25 games / iter
- 150 gradient steps / iter, batch size 64
- Sample with replacement
- value_weight=0.25

**Model**
- 3 residual blocks, 32 channels

**MCTS / self-play**
- 150 sims, c_puct=1.0

**Replay buffer**
- 30k FIFO, draw_keep_ratio=1.0

**Gating**
- 20 games, threshold 0.55

**Command**
```bash
python run.py \
  --iterations 600 \
  --buffer-size 30000 \
  --draw-keep-ratio 1.0 \
  --results-dir results/abl1_baseline
```

---

## Replay-buffer size sweep

Vary `--buffer-size`, everything else baseline. `buffer=30k` is the baseline — no separate run needed.

### Ablation 2 — buffer 10k
```bash
python run.py \
  --iterations 600 \
  --buffer-size 10000 \
  --draw-keep-ratio 1.0 \
  --results-dir results/abl2_buffer_10k
```

### Ablation 3 — buffer 3k
```bash
python run.py \
  --iterations 600 \
  --buffer-size 3000 \
  --draw-keep-ratio 1.0 \
  --results-dir results/abl3_buffer_3k
```

---

## Gating-games sweep

Vary `--gate-games`, everything else baseline. `gate-games=20` is the baseline — no separate run needed.

### Ablation 4 — gate 50
```bash
python run.py \
  --iterations 600 \
  --buffer-size 30000 \
  --draw-keep-ratio 1.0 \
  --gate-games 50 \
  --results-dir results/abl4_gate_50
```

### Ablation 5 — gate 120
```bash
python run.py \
  --iterations 600 \
  --buffer-size 30000 \
  --draw-keep-ratio 1.0 \
  --gate-games 120 \
  --results-dir results/abl5_gate_120
```

---

## Training-steps sweep

Vary `--steps-per-iter`, everything else baseline.

### Ablation 6 — 50 steps/iter
```bash
python run.py \
  --iterations 600 \
  --buffer-size 30000 \
  --draw-keep-ratio 1.0 \
  --steps-per-iter 50 \
  --results-dir results/abl6_steps_50
```

### Ablation 7 — 100 steps/iter
```bash
python run.py \
  --iterations 600 \
  --buffer-size 30000 \
  --draw-keep-ratio 1.0 \
  --steps-per-iter 100 \
  --results-dir results/abl7_steps_100
```
