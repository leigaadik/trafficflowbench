# Experiments

Local holdout: `fit` = train 前 8 个月（≤ 2031-01-31），`eval` = 2031-02 整月。
`src/` 保持只读，所有自有代码在 `mywork/`。

## 基线锚点

| 记录项 | 值 | 来源 |
|---|---:|---|
| 线上首次提交 `S_total` | **0.55343** | Kaggle（baseline，未改动） |
| 本地 holdout `S_state` | **0.7306** | `mywork/calibrate_t1.py`，十走廊平均 |

## Task 1 评估器标定（2026-09-22）

用三个已知输入检验评估器是否可信：

| 输入 | 本地 | 线上参照 | delta |
|---|---:|---:|---:|
| 全 0 | 0.0000 | 0.0000 | +0.0000 |
| baseline（前 8 月 fit） | 0.7306 | 0.6929 | **+0.0377** |
| 真值（未掩码层） | 1.0000 | 1.0000 | -0.0000 |

线上参照取自 `docs/BASELINES.md` 的 `S_state` 列（十走廊 validation 均值）。
注意别拿 `S_total` 的 0.5534 / 0.9945 来对照 Task 1——那是四任务总分。

**结论**：两端精确对齐，评估器逻辑正确。中间偏高 0.0377，说明 holdout 月
（2031-02）比 validation 月（2031-03）**更容易**——很可能因为 validation 带
5 起事件而 holdout 事件更少。这是数据分布差异，不是 bug。

**使用方式**：本地分数只用于**相对比较**，不能直接当作线上分数。

## 各走廊 baseline（holdout）

| Panel | S_state |
|---|---:|
| D12_I405_N | 0.8341 |
| D12_I405_S | 0.7919 |
| D12_I5_N | 0.6572 |
| D12_I5_S | 0.6359 |

（其余六条见 `reports/calibrate/`）

走廊间差异很大（0.63 ~ 0.83），说明**平均分会掩盖单条走廊的退化**，
后续改进要按 panel 看，不能只看一个数。

## 四任务离线评估能力总表（2026-09-22）

对照 `docs/BASELINES.md` 的十走廊 validation 均值：

| 任务 | 本地 | 线上 | delta | 结论 |
|---|---:|---:|---:|---|
| `S_state` (T1) | 0.7306 | 0.6929 | +0.0377 | ✅ 可用，两端标定精确落在 0 / 1.0 |
| `S_queue` (T2) | 0.3074 | 0.3017 | **+0.0057** | ✅ 高度可信 |
| `S_physics` (T3) | 0.3244 / 真值 0.9969 | 0.3549 / 真值 0.9636 | 判别幅度 0.673 vs 0.609 | ✅ 可行，见下 |
| `S_link` (T4 的 0.25) | 0.9999 | — | — | ⚠️ 已饱和 |
| `S_ODME` (T4) | 不可算 | 0.8359 | — | ❌ 无 OD 真值 |

### Task 2：可以放心用

`score_task2.py` 支持 `--truth-file`，而 train 的**未掩码层保留了 horizon 期间
的数据**（被抹空的只是 masked 视图），所以标签可重建：

```bash
python mywork/build_queue_truth.py --release-root data/kaggle_public \
    --output reports/submit/queue_targets_train.parquet
python src/task2/score_task2.py --submission <csv> --release-root data/kaggle_public \
    --split train --truth-file reports/submit/queue_targets_train.parquet
```

真值提交得 IoU 1.0，persistence 得 0.3074（线上 0.3017）。**偏差仅 0.006**。
唯一瑕疵：标签取自观测层而非 underlying state，且 `fd_parameters.csv` 没有
`v_cut` 列，退化为 `0.6 * free_speed`。两者对阈值附近的判定有轻微影响。

### Task 3：可行，靠反解净流入

**失败的第一条路**（记录以免重走）：用「拓扑 + 观测流量」推边界通量，真值
`S_LWR` 仍只有 0.0006。诊断显示 `sum|dN| = 544,503` 而 `sum|rhs| = 108,138,465`
——拓扑里 `incoming/outgoing` 混着 `CONN-*` 这类无观测的连接器，直接求和量级就
翻倍，再加上观测噪声，残差彻底淹没信号。

**可行的方法**：守恒式里真正的未知只有净流入，而 eval 月的 `N` 能从真值算出来，
于是可以反解：

```
dN_true = dt*(q_in + r_on - q_out - r_off)
⇒  q_in - q_out = dN_true/dt - r_on + r_off
```

把 `q_out` 钉在本 link 自己的流量上、修正量全给 `q_in`，得到的通量对真值精确
成立。它只由真值数据决定、与提交无关——和 `mainline_states` 作为 T1 真值是同一
性质，不是循环论证。

```bash
python mywork/build_boundary_flux.py --release-root data/kaggle_public \
    --output reports/submit/flux_conservation.parquet     # 默认 --outflow-mode conservation
python mywork/calibrate_t3.py --release-root data/kaggle_public \
    --boundary-flux reports/submit/flux_conservation.parquet \
    --submission truth <csv> --submission baseline <csv>
```

结果（D12_I5_N）：

| 输入 | 本地 | 线上 |
|---|---:|---:|
| baseline | 0.3244 | 0.3549 |
| 真值 | 0.9969 | 0.9636 |
| 判别幅度 | **0.673** | 0.609 |

**重要：优化时盯 `E_LWR`，不要盯 `S_LWR`。** `S_LWR = max(0, 1-min(1,E_LWR))`
在 `E_LWR > 1` 时恒为 0，存在死区。用真值/baseline 按 α 混合验证过：

| α（真值占比） | 0 | 0.25 | 0.50 | 0.75 | 0.90 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| `E_LWR` | 2.417 | 1.936 | 1.438 | 0.853 | 0.401 | 0.000 |
| `S_LWR` | 0 | 0 | 0 | 0.147 | 0.599 | 1.000 |

`E_LWR` 全区间连续单调；`S_LWR` 要 α≳0.6 才开始动。所以小幅改进应看
`E_LWR` 是否下降。

### Task 4：无需投入

`S_link` 已达 0.9999——baseline 的 nnls 目标就是 `min ||Af-c||²`，本来就在拟合
counts，这一项已饱和。`S_od` / `S_dev` / `S_attr` 需要 `base_od.csv`，而该文件
在数据包中不存在。加上 T4 总提升空间仅 0.033，性价比最低，建议不碰。

## 下一步

- [ ] 用本地 T1/T2 评估做第一个改进（T1 时空融合；T2 队列传播）
- [ ] 改进后提交一次，对齐「本地增量 → 线上增量」
- [ ] T3 靠 T1 改进间接带动，用榜单验证
