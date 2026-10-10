# Kronos A股5分钟微调实验交接文档

更新时间：2026-09-15

本文用于交接当前 Kronos LoRA/全参数微调研究，重点回答四个问题：

1. 哪些现象已经被实验验证；
2. 哪些实验可以直接比较，具体差异有多大；
3. 当前瓶颈更可能在哪里，下一步如何排优先级；
4. 如何利用 S1 top-5 覆盖率设计下一阶段预测与训练实验。

项目总体结构、完整命令和函数索引见仓库根目录
[`DEVELOPMENT_GUIDE.md`](../DEVELOPMENT_GUIDE.md)。本文只聚焦实验事实和后续研究设计。

---

## 1. 当前研究目标

Kronos 将每根标准化 K 线编码成两个层级 token：

```text
历史 S1/S2 token
    → 共享 Transformer context
    → head.proj_s1 → S1 分布
    → 选择/采样/边缘化 S1
    → Dependency cross-attention(context, S1 embedding)
    → head.proj_s2 → S2 分布
```

当前目标不是单纯降低 teacher-forced 总 loss，而是：

- 提高真实时间外验证集上的 S1 loss、top-1 和 top-5；
- 让 S2 在实际可用的预测 S1 条件下改善，而非只在真实 S1 条件下改善；
- 判断瓶颈来自 LoRA 容量、目标模块、数据异质性、时间漂移、Tokenizer 域偏移还是任务定义；
- 最终将 token 指标转化成下一根 K 线方向概率和解码价格指标。

---

## 2. 指标和比较规则

### 2.1 Loss

当前总目标为：

```text
loss = (w1 * s1_loss + w2 * s2_loss) / (w1 + w2)
```

S1-only 实验使用：

```text
w1=1.0, w2=0.0
```

此时日志仍可计算 S2 loss，但 S2 loss 不进入总目标，也不会产生有效优化贡献。

### 2.2 Top-k

- `s1_top1_accuracy`：真实 S1 是否为概率最高的候选；
- `s1_top5_accuracy`：真实 S1 是否位于概率最高的5个候选中；
- S2 指标含义相同；
- 所有指标只统计 `target_mask=True` 的 token 位置。

Top-5 是候选覆盖率，不等于模型知道 top-5 中哪一个是真实 token。推理时若没有额外选择器，
不能把 top-5 直接当成 top-1 使用。

### 2.3 不能直接比较的情况

只有以下条件都一致时，loss/top-k 才能做严格横向比较：

- 同一 `token_cache_path` 和数据指纹；
- 同一 train/validation/test 时间边界；
- 同一 `context_length`、`stride`；
- 同一 `target_mode`、`target_length`；
- 同一 split 和 target 数；
- 同一基座/Tokenizer；
- 同一 S2 conditioning。

当前实验至少分为三组：

| 组别 | 数据与目标 | 窗口/target | 用途 |
|---|---|---:|---|
| A | 全市场，`all512` | validation 49,360窗口、18,487,546 targets | S01～S06条件消融 |
| B | SW2半导体，`all512` | validation 1,088窗口、359,918 targets | 板块S1训练 |
| C | SW2半导体，`last240` | validation 5,536窗口、5,536 targets | 最后一位置基座/目标实验 |

组 A、B、C 的绝对 loss 不能互相比较。例如 `last` 每个窗口只统计一个目标，其难度和类别分布
可能与 `all` 完全不同。

---

## 3. 全市场基座与 LoRA 结果

### 3.1 基座完整条件消融

数据源：`base_s1_ablation_validation.json`。

```text
validation windows = 49,360
target count       = 18,487,546
s1 loss            = 2.496871
s1 top-1           = 27.1060%
s1 top-5           = 67.6539%
s1 entropy         = 2.513464
```

S2 条件消融：

| S1条件 | S2 loss | S2 top-1 | S2 top-5 |
|---|---:|---:|---:|
| teacher | 1.799266 | 43.3750% | 83.7264% |
| greedy | 1.799110 | 43.3788% | 83.7257% |
| sample | 1.799120 | 43.3792% | 83.7250% |
| shuffle | 1.799154 | 43.3803% | 83.7259% |
| random | 1.799377 | 43.3770% | 83.7212% |
| constant | 1.799255 | 43.3786% | 83.7236% |

**已确认结论**：基座 S2 几乎忽略 S1 条件。真实、预测、打乱、随机或常量 S1 都无法显著
改变 S2 指标，S2 主要由共享 Transformer context 决定。

### 3.2 LoRA 对 S1 的实际提升幅度

同一验证集上的短周期 LoRA（S01/S02/S04/S05/S06）稳定处于：

```text
s1 loss  ≈ 2.4677～2.4679
s1 top-1 ≈ 27.62%
s1 top-5 ≈ 68.32%
```

与基座相比约为：

```text
s1 loss 下降      ≈ 0.029
s1 top-1 提高     ≈ 0.51个百分点
s1 top-5 提高     ≈ 0.67个百分点
```

因此准确表述应是：

> LoRA 对全市场 S1 有稳定但很小的改善，随后进入平台；不能说完全没有学习，也没有证据说明
> 增加 rank 会把这种小幅改善扩大为显著收益。

### 3.3 高容量 `proj_s1`/联合训练

`qv_s1_r32_full_gradient_check` 的 teacher 评估：

```text
s1 loss  = 2.469757
s1 top-1 = 27.6006%
s1 top-5 = 68.2772%
s2 loss  = 0.823218
s2 top-1 = 66.0992%
s2 top-5 = 98.7294%
```

S2 teacher 指标大幅改善，但 S1 与低 rank 短周期结果基本相同。这说明扩大 S1 head 容量没有
在全市场验证集上带来对应收益；S2 的巨大改善来自使用真实 S1 的条件路径。

---

## 4. S01～S06 条件实验对比

### 4.1 实验定义

| 实验 | 训练条件 | Dependency是否训练 | 目的 |
|---|---|---:|---|
| S01 | greedy | 否，`self-attn-qv-s1` | 只改善共享路径/S1 |
| S02 | greedy | 是，`qv-s1` | 预测S1条件下训练Dependency |
| S03 | teacher | 是，`qv-s1` | 测试真实S1条件上限 |
| S04 | teacher | 否，`self-attn-qv-s1` | 隔离teacher与Dependency作用 |
| S05 | sample | 是，`qv-s1` | 提高错误S1鲁棒性 |
| S06 | 10% teacher + 90% sample | 是 | 折中条件收益与鲁棒性 |

### 4.2 主要结果

| 实验 | teacher S2 loss | greedy S2 loss | sample S2 loss | 解释 |
|---|---:|---:|---:|---|
| S01 | 1.779670 | 1.779452 | 1.779456 | 基本忽略S1 |
| S02 | 1.778898 | 1.778493 | 1.778530 | 基本忽略S1 |
| S03 | **0.865463** | 2.789797 | 3.245881 | 强依赖真实S1，暴露偏差严重 |
| S04 | 1.779300 | 1.779459 | 1.779462 | 不训练Dependency时teacher无作用 |
| S05 | 1.778490 | 1.778491 | 1.778490 | 对错误S1稳定，但重新学会忽略S1 |
| S06 | 1.647082 | 1.783852 | 1.785403 | 保留部分teacher收益，预测分支未改善 |

### 4.3 已确认结论

1. **Dependency 有能力使用 S1**：S03 teacher S2 loss 从约1.8降到0.865。
2. **错误 S1 会成为强负收益**：S03 greedy/sample 明显差于基座。
3. **不训练 Dependency 不会自然形成 S1→S2 关系**：S01/S04几乎不受S1条件变化影响。
4. **sample训练主要通过忽略S1获得鲁棒性**：S05所有条件结果接近。
5. **当前mixed权重没有解决核心问题**：S06恢复teacher收益，但predicted分支没有超过S05。
6. **teacher forcing不会导致S1本身不下降**：S1 logits在Dependency之前产生，S1 CE有独立梯度。

---

## 5. Top-k 概率边缘化结果

当前 `model.topk_marginalized_s2_log_probs()` 对当前模型 S1 分布的 top-k 候选逐个计算条件 S2，
再按归一化 S1 概率求和。S1 权重已 `detach`，S2 loss不会通过混合权重更新S1 predictor。

S06 adapter 的结果：

| K | temperature | S2 loss | S2 top-1 | S2 top-5 |
|---:|---:|---:|---:|---:|
| greedy(K=1) | — | 1.783852 | 43.5724% | 84.0208% |
| 5 | 1.0 | 1.780082 | 43.7320% | 84.0807% |
| 5 | 1.2 | 1.779989 | 43.7354% | 84.0815% |
| 10 | 1.0 | 1.779307 | 43.7428% | 84.0923% |
| 20 | 1.0 | **1.778901** | **43.7477%** | **84.0999%** |

Top-20 相比 greedy：

```text
S2 loss      -0.004951
S2 top-1     +0.1753个百分点
S2 top-5     +0.0791个百分点
```

**结论**：概率边缘化方向有效但收益很小。K增加时逐渐接近S05水平，没有明显超过S05的
`1.778490`。不建议继续只搜索K和temperature；下一步应该改变候选来源、概率校准或训练目标。

---

## 6. 梯度、容量与可学习性结论

### 6.1 小数据能够被记忆

S1-only小数据过拟合曾达到：

```text
epoch=100
train_s1=0.023969
same_set_s1=0.001528
s1_top1=100%
s1_entropy=0.0036
```

这证明以下链路正常：

- Token cache和shift对齐；
- target mask和S1 CE；
- LoRA注入与`proj_s1`；
- backward、梯度累积和optimizer更新。

因此真实验证集平台不是“模型完全无法记忆”或“梯度没有传播”。

### 6.2 未发现明确梯度冲突

共享 Transformer 上分离计算的 S1/S2 梯度余弦均为正，最低观察约0.02，后期常见：

```text
cos ≈ 0.19～0.28
s2/s1梯度范数比 ≈ 0.45～0.54
```

当前没有证据支持通过冻结 S2 的共享 Transformer 更新来消除强梯度对抗。

### 6.3 Rank不是唯一瓶颈

全市场和板块已测试 rank 4/8/16/32、提高 `proj_s1` rank、完整解冻 `proj_s1`，以及扩展
QKVO/fusion/FFN。高rank有助于小样本记忆，但没有稳定把真实验证S1 loss降到2.0以下。

因此：

- 不能排除某个特定结构需要更高rank；
- 但“训练量大 + rank太小”不是当前最符合证据的唯一解释；
- 后续容量实验必须保持数据、step和学习率一致，只改变一个变量。

---

## 7. SW2半导体板块实验

### 7.1 `all512`基座

数据源：`sw2_base_all512_validation.json`。

```text
validation windows = 1,088
targets            = 359,918
s1 loss            = 2.349023
s1 top-1           = 28.5490%
s1 top-5           = 71.7494%
```

### 7.2 `all512` LoRA 20 epoch

同一验证集上，S1-only r8实验：

```text
epoch 1  validation s1 loss = 2.335539, top-1=28.8635%, top-5=72.0075%
epoch 2  validation s1 loss = 2.327852, top-1=29.0108%, top-5=72.2053%
epoch 13 validation s1 loss = 2.320816, top-1=29.1500%, top-5=72.3301%  # best
epoch 20 validation s1 loss = 2.320941, top-1=29.1600%, top-5=72.3243%
```

相对基座的最佳改善：

```text
s1 loss  -0.028206
s1 top-1 +0.6010个百分点
s1 top-5 +0.5807个百分点
```

主要下降发生在前几个epoch；epoch13以后基本平台，末期cosine学习率降到0，继续训练没有
实际更新空间。

### 7.3 r4/r8和训练时长

同一板块 test 结果：

| 实验 | epoch完成 | S1 loss | top-1 | top-5 |
|---|---:|---:|---:|---:|
| r4 | 2 | 2.312926 | 29.2857% | 72.4082% |
| r8 | 5 | 2.305701 | 29.5486% | 72.5444% |
| r8 | 20 | 2.306669 | 29.5572% | 72.5215% |

r8长训练相对r4有小幅改善，但r8从约5到20 epoch没有继续改善top-5。由于rank和训练预算
同时变化，不能把全部差异归因于rank；只能确认继续延长当前cosine-to-zero训练收益很低。

### 7.4 `last240`基座不能与`all512`直接比较

```text
validation windows/targets = 5,536
s1 loss                    = 2.584606
s1 top-1                   = 27.0773%
s1 top-5                   = 65.8779%
```

该结果只监督最后一个位置，目标分布与`all512`不同。较高loss不能证明历史240比512差，必须
在同一target定义下分别评估240/512才可归因。

### 7.5 归档一致性问题

`sw2_all512_s128_qkvo_fusion_s1_r8_e20/` 的目录名称写着 `qkvo_fusion`，但其
`config.json` 实际记录：

```text
run_name       = sw2_all512_self_attn_qv_s1_r8_e20
target_modules = q_proj, v_proj, proj_s1
stride         = 24
```

因此该目录不能作为QKVO/fusion实验结果引用。实验身份必须以`config.json`、数据指纹和
run_directory为准，不能以外层复制目录名或图片文件名为准。

---

## 8. 已验证问题、已排除解释和未证实假设

### 8.1 已验证

1. 基座 S2 几乎忽略 S1。
2. Dependency 在真实 S1 条件下可以学到很强的 S1→S2 关系。
3. 预测 S1 错误会让强 Dependency 发生严重暴露偏差。
4. sample/mixed可以控制崩溃，但当前收益主要来自重新依赖context。
5. S1训练链路和小数据记忆能力正常。
6. 当前实验没有显示强S1/S2梯度冲突。
7. 全市场和板块LoRA对S1都有约0.5～0.7个百分点的小幅top-k提升，然后平台。
8. 单纯增加rank、延长cosine-to-zero训练或只搜索top-k不能解决主要瓶颈。

### 8.2 已排除或应降低优先级

- “teacher forcing阻止S1学习”：不成立，teacher直接影响S2条件路径。
- “梯度完全没有传播”：小数据100% top-1已否定。
- “S2一定在对抗S1梯度”：当前余弦为正，没有支持证据。
- “只要把proj_s1 rank提高就能解决”：真实验证没有对应收益。
- “top-5为68%，随机从五个里选就能提高”：错误；随机选择的理论命中约只有13.6%。

### 8.3 仍需验证

1. 不同行情阶段造成时间分布漂移；
2. 全市场股票/行业/波动率异质性稀释梯度；
3. 固定Tokenizer在A股5m上的重建误差或码本利用不理想；
4. S1标签本身条件熵较高、类别不平衡；
5. 全序列token目标与“下一根K线/方向”业务目标不一致；
6. 高重叠窗口增加step但没有增加足够独立信息；
7. 预训练时间embedding不适配A股交易时段，而LoRA profile没有更新它；
8. 当前时间特征缺少year、实际时间间隔、交易阶段和显式下一目标时间。

---

## 9. 如何利用 S1 top-5 进行下一步实验

### 9.1 先明确 top-5 能提供什么

全市场基座：

```text
top-1 = 27.11%
top-5 = 67.65%
```

说明约40.55%的位置属于：

```text
top-1错误，但真实token仍在top-5候选内
```

这是可利用空间，但需要解决“候选选择或概率整合”。不能在推理时使用真实标签挑出正确候选。

### 9.2 P0：候选覆盖与互补性报告

在修改训练前，先缓存基座和LoRA对同一验证位置的S1 logits，计算：

```text
base top-1/top-5/top-10/top-20 coverage
LoRA top-1/top-5/top-10/top-20 coverage
base top-5 ∪ LoRA top-5 的oracle coverage
两者top-5 Jaccard overlap
真实S1只在base候选、只在LoRA候选、两者都有、两者都无的比例
top1-top2 probability margin
预测熵与top-k命中率的关系
```

目的：判断两个模型是否真正互补。如果两组top-5几乎完全相同，简单ensemble上限很低；如果
union覆盖显著高于单模型，就值得开发混合候选。

### 9.3 P1：无需重新训练的S1概率融合

对基座和LoRA的S1分布先做温度校准，再融合：

```python
base_p = softmax(base_logits / base_temperature)
lora_p = softmax(lora_logits / lora_temperature)
mixed_p = base_weight * base_p + (1.0 - base_weight) * lora_p
```

也可测试log概率几何融合：

```text
log p_mix ∝ λ log p_base + (1-λ) log p_lora
```

建议网格：

```text
base_weight = 0.00, 0.25, 0.50, 0.75, 1.00
temperature = 0.8, 1.0, 1.2, 1.5
K           = 1, 5, 10, 20
```

从融合分布取top-k，再使用同一个S2模型执行：

```text
p(s2|context) = Σ_i normalized_p_mix(s1_i) * p(s2|context,s1_i)
```

第一轮优先比较：

| S1候选来源 | S2条件模型 | 目的 |
|---|---|---|
| base | S05 | 基座候选 + 鲁棒S2 |
| LoRA | S05 | 当前稳定基线 |
| base/LoRA融合 | S05 | 判断候选互补是否转化为S2收益 |
| base/LoRA融合 | S06 | 判断部分条件敏感性是否有用 |
| base/LoRA融合 | S03 | 仅作上限/风险诊断，预计对错候选很敏感 |

当前代码只支持“同一模型产生S1并计算S2”，尚不支持跨模型logits融合。新增实现应放在独立评估
入口，先不要改训练主循环。

### 9.4 P2：置信度门控

并非所有位置都需要top-k。可根据S1置信度选择：

```text
若 top1_probability >= τ 且 top1-top2 margin >= m：使用greedy
否则：使用top-5/10概率边缘化
```

优点：

- 高置信位置保持低延迟；
- 高熵位置才支付K次Dependency计算；
- 避免在本来正确的高置信top-1上引入多候选噪声。

候选阈值应在校准集选择，在保留验证/测试集报告，不能在同一测试集反复调参。

### 9.5 P3：Oracle诊断，不能部署

为确定top-5候选是否对S2真的有价值，可做两个oracle实验：

1. 若真实S1在top-5中，强制选择真实S1；否则选择top-1；
2. 在top-5候选中选择使真实S2 loss最小的候选。

两者只用于估计上限：

- oracle提升很大：候选集足够，缺少的是选择器/校准；
- oracle提升很小：即使top-5覆盖高，现有Dependency也无法将候选转成有用S2；
- oracle与实际边缘化差距大：应优化候选权重或训练S2的候选鲁棒性。

任何oracle结果都不能作为部署指标。

### 9.6 P4：候选边缘化训练

若P0～P3证明候选有潜力，再训练：

```text
L = w1 * CE(s1)
  + w2 * [-log Σ_i p(s1_i|context) p(s2_target|context,s1_i)]
```

当前top-k实现detach了`p(s1_i)`。应做严格对照：

| 版本 | S2梯度是否进入S1概率 | 风险 |
|---|---:|---|
| detached | 否 | 稳定，但S2不能帮助重排S1 |
| joint | 是 | 可学习任务相关排序，但可能走捷径/坍缩 |

Joint版本必须保留独立S1 CE，并监控S1 top-1/top-5、熵和候选覆盖，防止模型仅为了S2而扭曲
S1分布。

### 9.7 P5：Beam/joint token预测

若最终目标是输出概率最高的一条K线，可保留多个联合候选：

```text
score(s1_i, s2_j)
  = log p(s1_i|context)
  + log p(s2_j|context,s1_i)
```

每一步保留联合分数最高的beam，再解码完整S1/S2序列。它比单纯先argmax S1更符合层级token
概率，但计算量随beam宽度增加。第一阶段只做下一根K线，不直接扩展到长轨迹。

### 9.8 Top-5实验必须同时评价连续K线

除token指标外，至少增加：

```text
下一根close方向准确率
up/down/flat概率的NLL或Brier score
OHLC MAE/相对误差
预测区间覆盖率
按月、股票、日内时段和波动率分组指标
推理延迟与显存
```

top-5 token提升不一定对应价格或交易信号提升。

---

## 10. 时间问题的下一步验证

当前时间处理：

- 按symbol内timestamp排序并做时间前向切分；
- 输入minute/hour/weekday/day/month embedding；
- 没有year、elapsed time、交易阶段或市场状态；
- 模型预测下一token时只看到输入token对应时间，不显式看到第一个目标时间；
- LoRA模式冻结预训练time embedding，全参数模式才会更新；
- 所有训练窗口等权，没有近期样本加权。

建议按同一模型输出：

1. 每月/每周S1 loss、top-1、top-5；
2. 开盘、上午、午后、尾盘分时段指标；
3. 连续5分钟、跨午休、跨日、存在缺失bar的分组指标；
4. 最近1/3/6/12个月训练、相同optimizer step、同一未来验证集；
5. LoRA与低学习率Kronos-small全参数短实验。

若同一时期保留集改善而未来月份不改善，支持时间漂移；若两者都不改善，应优先检查Tokenizer
和S1标签条件熵。

---

## 11. 下一阶段优先级

### P0：修复实验归档和建立可比基线

- 每个run完整保存config、dataset summary、history、metrics、result、best/last和plots；
- 禁止用外层复制目录名判断实验配置；
- 为组A/B/C分别建立base/LoRA/full基准表；
- 加入最后有效target位置和按时间分组指标。

### P1：完成S1候选覆盖/互补性报告

这是利用top-5前必须完成的最低成本实验。没有union coverage和候选重叠率，就无法判断基座与
LoRA融合是否值得开发。

### P2：实现独立的跨模型S1融合评估

先只做验证入口，不修改训练。比较base、LoRA、线性融合、几何融合和置信门控，再将候选送入
S05/S06的S2路径。

### P3：运行时间与数据分组诊断

按月份、交易时段、股票和波动率定位平台来源。避免继续只看一个全局平均loss。

### P4：Kronos-small全参数短实验

使用同一cache/目标和低学习率`5e-6～1e-5`，与LoRA在相同optimizer step下比较。目的是判断
冻结embedding、time embedding、norm等是否为瓶颈，不是直接投入长周期训练。

### P5：只有候选实验显示上限后再做joint top-k训练

先确认oracle和融合有收益，再决定是否让S2边缘化loss反向进入S1概率。否则会增加复杂度但
无法回答核心问题。

---

## 12. 建议的成功标准

不要因为单次loss下降几个千分点就宣布成功。建议至少满足：

1. 同一保留验证集 predicted-S2 loss 稳定优于S05/S06 greedy基线；
2. S1 top-5不下降，top-1或概率校准至少一项改善；
3. 改善在多个时间段/随机种子上方向一致；
4. teacher指标与predicted指标同时保存，不能通过完全忽略S1缩小gap；
5. 解码后的方向概率或OHLC指标同步改善；
6. 推理成本可以接受。

以当前top-20相对greedy的提升为参照，下一方案若不能稳定超过：

```text
S2 loss下降 0.005 左右
或 S2 top-1提高 0.2 个百分点左右
```

则暂不值得增加复杂部署路径。该阈值是工程筛选基准，不是统计显著性结论；正式结论应报告
bootstrap置信区间或按时间块重复结果。

---

## 13. 交接检查清单

下一位开发人员开始实验前应确认：

1. `config.json`而不是目录名决定实验身份；
2. 所有对比使用相同cache、split、target mask和target count；
3. 验证同时运行teacher、greedy、sample、shuffle，必要时top-k；
4. S1 top-5只视为候选覆盖率，不作为可部署准确率；
5. top-k搜索在校准集完成，测试集只运行一次；
6. rank实验保持训练step和学习率一致；
7. `resume-from`与`initialize-from-adapter`不混用；
8. 全参数checkpoint使用`evaluate_base`加载；
9. 不把小数据过拟合结果解释为泛化能力；
10. 不把不同target mode的绝对loss放在同一表中归因；
11. 记录时间边界、成分股快照和幸存者偏差；
12. 最终报告token、连续价格、方向概率和推理成本四类指标。

---

## 14. 当前总体判断

现有证据显示，训练代码可以学习和记忆，LoRA也能给S1带来小幅稳定改善；真正问题是这种改善
在真实验证集上很快平台，而S2要么忽略S1保持稳定，要么在teacher条件下过度依赖真实S1并在
推理时崩溃。

S1 top-5约68%～72%提供了候选空间，但不是现成答案。下一步最有信息量的工作不是继续放大
rank，而是：

```text
候选互补性统计
  → base/LoRA概率校准与融合
  → 置信门控/top-k边缘化
  → oracle上限判断
  → 必要时joint top-k训练
```

同时开展按月份和交易时段的时间诊断，以及同预算的Kronos-small全参数短实验。只有这些结果
确认候选或完整表示确有可利用增益后，才值得进入长周期训练和实时部署。

---

## 15. 实验目录与文件命名规则

### 15.1 命名的基本原则

1. 目录名用于快速识别，`config.json` 才是实验身份的唯一事实来源。
2. `--run-name` 必须以字母或数字开头，只使用字母、数字、`.`、`_`、`-`。推荐全部使用小写 ASCII，避免服务器、Shell 和前端的编码差异。
3. 日期用 `YYYYMMDD`，时间用 `YYYYMMDD_HHMMSS`；小数在文件名中用 `p` 代替小数点，例如温度 1.2 写成 `t1p2`。
4. 同一组对照只改一个实验因子，其他字段保持完全相同。
5. 不把结果 JSON 或图片单独拷出后重命名为“最新”。应保留整个 run，或者让拷贝名包含原 run name。
6. 显式的 `--run-name` 已存在时训练会拒绝覆盖。不要手工删除旧 run 来复用名字。

### 15.2 数据范围标识

| 范围 | 推荐 ID | 说明 |
|---|---|---|
| A 股全市场 | `a_share_5m` | 应在 metadata 中另外记录是否包含北交所 |
| 申万二级半导体 | `sw2_semiconductor` | 板块类型和名称同时体现，不只写 `semiconductor` |
| 多板块并集 | `sw2semi_gnpower_union` | 顺序固定，详细来源以 `sector_membership.json` 为准 |
| 单股 | `stock_688449_sh` | 文件名使用小写后缀，实际 XtQuant 代码仍是 `688449.SH` |

板块成分股是快照。正式实验推荐把快照日期写入目录：

```text
data/sectors/sw2_semiconductor/universe/20260915/
```

### 15.3 Bucket 与 Token Cache 命名

推荐结构：

```text
data/<scope-id>/
├── universe/<snapshot-date>/
├── raw_5m/
├── buckets/b<bucket-count>_<source-id>/
└── token_cache/
    └── c<context>_s<stride>_<target>_tr<train-end>_va<validation-end>_tok<tokenizer-id>_v<format>/
```

示例：

```text
data/sectors/sw2_semiconductor/
├── universe/20260915/
├── raw_5m/
├── buckets/b16_src20260915/
└── token_cache/c512_s128_all_tr20260630_va20260731_tokbase_v1/
```

`target` 的编码为 `all`、`last` 或 `tail280`。缓存身份还必须核对 manifest 里的日期边界、Tokenizer 指纹、样本数和源数据指纹，不能只靠目录名。

### 15.4 Run name 标准

推荐顺序：

```text
<scope>_<base>_<objective>_<target>_c<C>s<S>_<profile>_r<R>a<A>_<condition>_eb<effective-batch>_lr<LR>_wu<W>_e<E>_sd<seed>
```

常用缩写：

| 字段 | 缩写 |
|---|---|
| Kronos-base / small | `base` / `small` |
| S1-only / S2-only / 联合 | `s1` / `s2` / `joint` |
| teacher / greedy / sample | `tch` / `grd` / `smp` |
| top-k, k=20, temperature=1.0 | `tk20t1p0` |
| mixed teacher 0.1 + predicted 0.9 | `mix_t10p90` |
| `self-attn-qv-s1` | `saqvs1` |
| `self-attn-qkvo-fusion-s1` | `saqkvofus1` |
| `self-attn-qkvo-fusion-top4-ffn-s1` | `saqkvofust4ffns1` |
| `self-attn-qkvo-fusion-ffn-s1` | `saqkvoffns1` |
| 全参数微调 | `full` |

示例：

```text
sw2semi_base_s1_tail280_c512s128_saqkvofus1_r8a16_eb16_lr1e-4_wu500_e20_sd100
sw2semi_small_joint_all_c512s128_saqvs1_r16a32_mix_t10p90_eb16_lr5e-5_wu500_e6_sd100
stock688449_small_s1_tail282_c512s128_full_eb8_lr1e-5_wu200_e20_sd100
```

有效 batch 按下式记录：

```text
effective_batch = batch_size_per_gpu × gradient_accumulation_steps × world_size
```

`eb` 只是用于阅读的摘要，每卡 batch、累积步数和 world size 仍必须保存在 `config.json`。

### 15.5 Run 内文件命名

run 目录内的标准名字不要人工改动：

```text
checkpoints/<experiment-root>/runs/<run-name>/
├── config.json
├── dataset_summary.json
├── metrics.jsonl
├── history.json
├── result.json
├── best_model/
├── last_model/
├── resume/
│   ├── latest.json
│   ├── step-02000/
│   └── step-04000/
└── plots/
    ├── loss.png
    ├── component_loss.png
    ├── token_metrics.png
    ├── gradient_norms.png
    ├── gradient_conflict.png
    └── learning_rate.png
```

独立评估文件推荐名称：

```text
eval_<split>_<condition>_<selection>[_k<K>_t<T>].json
ablation_<axis>_<split>_<selection>.json
```

例如 `eval_validation_both_best.json`、`eval_validation_topk_best_k20_t1p0.json`和
`ablation_s1_condition_validation_best.json`。评估文件中必须再保存 adapter/base 绝对路径、模型选择、cache 指纹和 target count。

---

## 16. 当前可调参数和实验方向全表

### 16.1 训练 CLI 直接可调参数

| 类别 | 参数 | 当前默认/选项 | 作用和注意事项 |
|---|---|---|---|
| 模型 | `--base-model-path` | `pretrained/Kronos-base` | 可换 Kronos-base/small；需与 adapter 身份匹配 |
| 模型 | `--tokenizer-path` | `pretrained/Kronos-Tokenizer-base` | 原 Parquet 训练时使用；Token Cache 必须核对指纹 |
| 输出 | `--adapter-output-dir` | Kronos-base LoRA 根目录 | 更换基座时建议换独立根目录 |
| 数据 | `--data-path` | `data/lora` | 原 Parquet 入口 |
| 数据 | `--token-cache-path` | 空 | 设置后优先读预 Tokenize shard |
| 切分 | `--train-end` / `--validation-end` | 空 | 建议总是显式设置；Cache 模式需与 manifest 一致 |
| 窗口 | `--context-length` | 512 | 最大 512；每个原始样本需要 `context+1` 行 |
| 窗口 | `--stride` | 训练 CLI 默认 32 | 控制窗口重叠和样本量；必须与 Cache manifest 一致 |
| 目标 | `--target-mode` | `all` | `all` / `last` / `tail` |
| 目标 | `--target-length` | 空 | 仅 `tail` 有效，范围 `1..context_length` |
| 方式 | `--finetune-mode` | `lora` | `lora` 或 `full` |
| 方式 | `--epochs` | 10 | 必须同时报告 optimizer step，不只比 epoch |
| 优化 | `--batch-size` | 8 | 每张 GPU 的 micro-batch |
| 优化 | `--gradient-accumulation-steps` | 1 | 达到累积次数才执行一次 `optimizer.step()` |
| 优化 | `--learning-rate` | `1e-4` | 配合 warmup 和 cosine decay |
| 优化 | `--warmup-steps` | 0 | 按 optimizer step 线性升温，之后 cosine decay |
| Loss | `--s1-loss-weight` / `--s2-loss-weight` | 0.5 / 0.5 | 内部按权重和归一化；`1/1` 与 `0.5/0.5` 等价 |
| S2 条件 | `--training-s2-conditioning` | `teacher` | `teacher`/`greedy`/`sample`/`topk`/三种 `mixed-*` |
| S2 混合 | `--s2-teacher-loss-weight` | 0.1 | 仅 `mixed-*` 分支使用 |
| S2 混合 | `--s2-predicted-loss-weight` | 0.9 | 与 teacher 权重归一化 |
| Top-k | `--s2-top-k` | 5 | S1 候选数，不是评估中 S2 top-5 指标的 5 |
| Top-k | `--s2-temperature` | 1.0 | 调整 S1 候选分布尖锐度 |
| LoRA | `--lora-target-profile` | `qv` | 决定添加 B/A 矩阵的模块集合 |
| LoRA | `--lora-r` / `--lora-alpha` | 8 / 16 | 全局 rank 和缩放；保持 `alpha/r` 可做纯容量对照 |
| S1 头 | `--proj-s1-rank` / `--proj-s1-alpha` | 空 | 需 profile 包含 `proj_s1` |
| S1 头 | `--full-finetune-proj-s1` | 关 | 完整解冻 `head.proj_s1`，不能同时设 `proj-s1-rank` |
| Dependency | `--dependency-lora-r` / `--dependency-lora-alpha` | 空 | 只覆盖 `dep_layer.cross_attn` Q/V；profile 不能排除 Dependency |
| Fusion | `--fusion-lora-r` / `--fusion-lora-alpha` | 空 | 需 profile 包含 `embedding.fusion_proj` |
| FFN | `--ffn-lora-r` / `--ffn-lora-alpha` | 空 | 需 profile 包含 FFN `w1/w2/w3` |
| 数据加载 | `--num-workers` | 0 | 可提高到 4/8/16 实测，过大可增加 RAM 和随机 IO |
| 诊断 | `--gradient-log-interval` | 50 | 记录裁剪前梯度范数；0 关闭 |
| 诊断 | `--gradient-conflict-interval` | 0 | 额外重放一个累积组计算 S1/S2 余弦；明显增加开销 |
| 验证 | `--validation-interval` | 2000 | 按 optimizer step 做全验证集评估；0 只保留 epoch-end |
| 保存 | `--checkpoint-interval` | 2000 | 保存可完整恢复的 `resume/step-*` |
| 早停 | `--early-stopping-patience` | 3 | 只按 epoch-end 的“未改善 epoch”计数 |
| 早停 | `--early-stopping-min-delta` | 0 | 重置 patience 所需的最小 Loss 下降；`best_model` 仍记录严格更低值 |
| 身份 | `--run-name` | 自动时间戳 | 建议使用本文命名规则 |
| 前端 | `--model-display-name` | 空 | 只是展示标签，不参与模型构建 |
| 恢复 | `--resume-from` | 空 | 恢复 adapter/full weights、optimizer、scheduler、step、epoch 和 RNG |
| 初始化 | `--initialize-from-adapter` | 空 | 只加载 LoRA/已保存模块，新建 optimizer、scheduler、计数器和 run |
| 初始化 | `--initialize-model-selection` | `best` | 当初始化路径是 run 时选 `best` 或 `last` |
| 设备 | `--device` | `auto` | 单卡可设 `cuda:0`；DDP 由 `torchrun` 传入 local rank |
| 精度 | `--precision` | `auto` | `auto`/`bf16`/`fp32`；Blackwell 正式训练建议 `bf16` |

LoRA profile 的当前可选值是：

```text
qv
qv-s1
self-attn-qv-s1
self-attn-qkvo-fusion-s1
self-attn-qkvo-fusion-top4-ffn-s1
self-attn-qkvo-fusion-ffn-s1
```

具体命中范围：

| Profile | 主要目标模块 | Dependency cross-attention |
|---|---|---|
| `qv` | 所有后缀为 `q_proj`/`v_proj` 的 Linear | 包含 |
| `qv-s1` | `qv` + `head.proj_s1` | 包含 |
| `self-attn-qv-s1` | 共享 Transformer Q/V + `head.proj_s1` | 明确排除 Dependency Q/V |
| `self-attn-qkvo-fusion-s1` | 共享 Transformer Q/K/V/O + `embedding.fusion_proj` + `head.proj_s1` | 排除 Dependency Q/K/V/O |
| `self-attn-qkvo-fusion-top4-ffn-s1` | 上一行 + Transformer 最后 4 层 FFN `w1/w2/w3` | 排除 Dependency Q/K/V/O |
| `self-attn-qkvo-fusion-ffn-s1` | 上述 attention/fusion/S1 + 所有 Transformer FFN | 排除 Dependency Q/K/V/O |

`--dependency-lora-r` 只能与包含 Dependency Q/V 的 profile 配合，因此不能直接用在名称以 `self-attn-` 开头的当前 profile 上。

### 16.2 配置类可调、但当前未暴露为训练 CLI 的参数

| 位置 | 参数 | 默认 | 说明 |
|---|---|---|---|
| `TrainingConfig` | `weight_decay` | 0.01 | AdamW 权重衰减 |
| `TrainingConfig` | `adam_beta1` / `adam_beta2` | 0.9 / 0.95 | AdamW 动量估计 |
| `TrainingConfig` | `max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `TrainingConfig` | `log_interval` | 50 | 基础训练日志间隔 |
| `TrainingConfig` | `seed` | 100 | 实验随机种子 |
| `QVLoRAConfig` | `dropout` | 0.05 | LoRA dropout |
| `QVLoRAConfig` | `bias` | `none` | PEFT bias 训练策略 |
| `DataConfig` | `clip` | 5.0 | 窗口归一化后的裁剪阈值 |
| `DataConfig` | 列名与特征列 | OHLCVA + 时间列 | 更改会影响数据协议和 Token Cache |
| `DataConfig` | `train_ratio` / `validation_ratio` | 0.8 / 0.1 | 仅未提供显式日期边界时作为开发回退 |

若需要系统扫描这些参数，应先把它们显式暴露到 CLI 并写入 `config.json`，不要为每次实验手工修改源码。

### 16.3 参数变更是否需要重建数据

| 变更 | 重建 Bucket | 重建 Token Cache | 说明 |
|---|---:|---:|---|
| rank/alpha/profile/Loss 权重/学习率/batch/epoch | 否 | 否 | 新建 run 即可 |
| base ↔ small | 否 | 通常否 | 前提是使用相同 Tokenizer 且 vocab 协议兼容 |
| `all` Cache → `last`/`tail` | 否 | 否 | 加载时收窄 `loss_mask` |
| `last`/`tail` Cache → `all` | 否 | 是 | 已删除的监督位置不能恢复 |
| 改 `context-length` 或 `stride` | 否 | 是 | 窗口索引发生变化 |
| 改 split 日期边界 | 否 | 是 | shard 已按 split 物化 |
| 改 Tokenizer 权重/版本 | 否 | 是 | 指纹和 token 标签变化 |
| 改 `clip`、OHLCVA 处理或特征协议 | 否 | 是 | Tokenizer 输入变化 |
| 改股票池 | 是 | 是 | 或从原 Parquet 重新过滤到独立数据目录 |
| 只改 bucket-count | 是 | 是 | bucket 是 Token shard 的输入组织 |
| 只改 samples-per-shard | 否 | 是 | token 语义不变，但 shard 物理布局变化 |

注意当前存在三个 stride 默认值：`DataConfig=24`、训练 CLI `=32`、
`build_shards tokenize=128`。这是一个已知易错点。正式命令中必须显式传入
`--context-length`、`--stride`、`--target-mode` 和必要的 `--target-length`，并与 manifest 核对。

### 16.4 当前可实验的优化方向

| 方向 | 主要控制项 | 回答的问题 |
|---|---|---|
| 数据粒度 | 全市场/板块/单股、近 1/3/6/12 月 | 异质性和时间漂移是否掩盖信号 |
| 监督密度 | `all`/`tail N`/`last`、stride | 窗口内早期 token 是否带来旧行情噪声 |
| LoRA 容量 | rank 4/8/16/32，保持 alpha/r | 是否真的受低秩容量限制 |
| 目标模块 | QV → QKVO+fusion → top4 FFN → all FFN | S1 需要改动多少表示层 |
| S1 头 | proj_s1 LoRA 独立 rank 或完整解冻 | 输出头是否是瓶颈 |
| Dependency | Dependency 独立 rank | S2 条件路径容量是否不足 |
| 任务配比 | S1/S2 Loss 权重、S1-only 预训练再 joint | 共享 Transformer 的优化重心是否偏向 S2 |
| S2 鲁棒性 | teacher/greedy/sample/top-k/mixed | 如何降低真实 S1 错误对 S2 的破坏 |
| 优化过程 | LR、warmup、有效 batch、epoch | 平台是欠优化还是表示上限 |
| 冻结范围 | LoRA vs `full-finetune-proj-s1` vs `full` | 冻结 embedding/time/norm 是否形成瓶颈 |
| 候选融合 | base/LoRA S1 top-5 union、概率融合、门控 | top-5 候选空间能否转化为更好 S2/价格预测 |
| 时间特征 | 按月/时段评估，年份/市场阶段扩展 | 当前时间 embedding 是否不足 |

实验优先级仍是先补齐评估和可比性，再扩容。小数据 100% top-1 已证明代码有记忆能力；因此长周期盲目把 rank 从 8 提到 32 的信息量低于“同预算 profile 对照 + 候选互补统计 + 时间分组”。

---

## 17. 股票池导出、全市场复用和单股数据

### 17.1 查找并固定板块股票池

MiniQMT 需已启动并登录。首先查询 XtQuant 实际提供的精确板块名：

```bash
python -m data_download.xtquant.sector_universe list --contains 半导体
```

导出一个板块快照：

```bash
python -m data_download.xtquant.sector_universe export \
  --sector-name SW2半导体 \
  --output-dir data/sectors/sw2_semiconductor/universe/20260915
```

重复 `--sector-name` 会生成多板块并集，股票代码自动规范化和去重：

```bash
python -m data_download.xtquant.sector_universe export \
  --sector-name SW2半导体 \
  --sector-name GN功率半导体 \
  --output-dir data/sectors/sw2semi_gnpower_union/universe/20260915
```

输出中 `symbols.txt` 是后续命令的输入，`sector_membership.json` 记录每只股来自哪些板块，`sector_metadata.json` 记录数量、生成时间和 SHA-256。这是生成时的当前成分股，不是历史逐日成分股，存在幸存者偏差。

`sector_universe` 的全部 CLI 控制项为：`list|export`、可重复的
`--sector-name`、`--output-dir`、`--contains`、`--exclude-beijing` 和
`--overwrite`。`--contains` 只负责列表筛选，导出时仍必须使用精确的
`--sector-name`。正式快照推荐新建日期目录，而不是使用 `--overwrite`。

### 17.2 Plan 和直接下载

先生成计划并检查股票数、batch 数和股票池哈希：

```bash
python -m data_download.xtquant.cli plan \
  --start-date 20250915 \
  --end-date 20260915 \
  --symbols-file data/sectors/sw2_semiconductor/universe/20260915/symbols.txt \
  --download-batch-size 50 \
  --output-dir data/sectors/sw2_semiconductor/raw_5m
```

确认后执行：

```bash
python -m data_download.xtquant.cli download \
  --start-date 20250915 \
  --end-date 20260915 \
  --symbols-file data/sectors/sw2_semiconductor/universe/20260915/symbols.txt \
  --download-batch-size 50 \
  --read-batch-size 50 \
  --retry-count 3 \
  --dividend-type front \
  --output-dir data/sectors/sw2_semiconductor/raw_5m
```

可重复传入多个 `--symbols-file`，`plan`、`download` 和 `filter-existing` 都会对所有文件的并集执行同一套清洗、A 股后缀过滤和去重。

下载 CLI 的其他可调项包括：

- `--years 1`：没有显式日期时的回溯年数；
- `--exclude-beijing`：排除北交所；
- `--include-today`：包含当日；
- `--no-resume`：关闭下载 manifest 断点续传；
- `--log-level`：日志级别；
- `retry-failed`：使用已保存 universe 重试失败批次，不要再传股票池参数。

板块下载不能直接写入已建立 manifest 的全市场下载目录，否则 universe SHA-256 和批次索引会冲突。

### 17.3 从已下载全市场 Parquet 离线过滤

全市场原始数据已存在时，不需要再请求 XtQuant：

```bash
python -m data_download.xtquant.cli filter-existing \
  --source-dir data/lora \
  --symbols-file data/sectors/sw2_semiconductor/universe/20260915/symbols.txt \
  --output-dir data/sectors/sw2_semiconductor/raw_5m
```

`filter-existing` 逐个读原 Parquet，保留目标股票并继续使用 `year=YYYY/month=MM` 分区。它工作在原始 K 线层，不能对已生成的 Token shard 做股票过滤。源目录和输出目录必须独立且不能互相嵌套。

完成后校验：

```bash
python -m data_download.xtquant.cli validate \
  --output-dir data/sectors/sw2_semiconductor/raw_5m
```

### 17.4 单股流程

单股参数与 `--symbols-file` 互斥。例如 688449.SH：

```bash
python -m data_download.xtquant.cli plan \
  --symbol 688449.SH \
  --start-date 20250915 \
  --end-date 20260915 \
  --output-dir data/stocks/688449_sh/raw_5m

python -m data_download.xtquant.cli download \
  --symbol 688449.SH \
  --start-date 20250915 \
  --end-date 20260915 \
  --output-dir data/stocks/688449_sh/raw_5m
```

也可从全市场数据直接过滤：

```bash
python -m data_download.xtquant.cli filter-existing \
  --source-dir data/lora \
  --symbol 688449.SH \
  --output-dir data/stocks/688449_sh/raw_5m
```

---

## 18. Parquet 分桶和 Token shard 分片

### 18.1 为什么先分桶

原始数据通常按时间分区，同一股票会出现在多个月份 Parquet 中。`bucketize` 逐文件读取，按 symbol 的稳定哈希将行写入固定 bucket。结果是：

- 同一股票在不同月份的数据一定进入同一个 bucket；
- 不同股票可在同一 bucket，但后续仍按 symbol 独立排序和切窗；
- 任何一个训练窗口都不会跨越股票代码；
- 无需一次将全市场数据 `concat` 到内存。

### 18.2 第一阶段：分桶

全市场参考命令：

```bash
python -m finetune_lora.build_shards bucketize \
  --data-path data/lora \
  --bucket-dir data/a_share_5m/buckets/b64_src20260915 \
  --bucket-count 64
```

板块参考命令：

```bash
python -m finetune_lora.build_shards bucketize \
  --data-path data/sectors/sw2_semiconductor/raw_5m \
  --bucket-dir data/sectors/sw2_semiconductor/buckets/b16_src20260915 \
  --bucket-count 16
```

单股可以用 1 个 bucket，或者从更大原始目录中在分桶时限定 symbol：

```bash
python -m finetune_lora.build_shards bucketize \
  --data-path data/lora \
  --bucket-dir data/stocks/688449_sh/buckets/b1_src20260915 \
  --bucket-count 1 \
  --symbol 688449.SH
```

经验起点是全市场 64 桶、中等板块 8‑16 桶、单股 1 桶。桶数不改变数据语义，只改变 IO 粒度和中间文件数。输出目录必须为空，不要向旧 bucket 目录增量混入新股票池。

### 18.3 第二阶段：Tokenize 并生成 shard

为了最大化后续可复用性，优先生成 `all` Cache：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.build_shards tokenize \
  --bucket-dir data/sectors/sw2_semiconductor/buckets/b16_src20260915 \
  --token-cache-path data/sectors/sw2_semiconductor/token_cache/c512_s128_all_tr20260630_va20260731_tokbase_v1 \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --train-end "2026-06-30 15:00:00" \
  --validation-end "2026-07-31 15:00:00" \
  --context-length 512 \
  --stride 128 \
  --target-mode all \
  --encode-batch-size 32 \
  --samples-per-shard 4096 \
  --device cuda:0
```

`tokenize` 的可调项是：

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--tokenizer-path` | 本地 Kronos-Tokenizer-base | 写入 Tokenizer 指纹 |
| `--train-end` / `--validation-end` | 必填 | 按 target timestamp 生成 train/validation/test mask |
| `--symbol` | 空 | 只 Tokenize 桶中某一股票 |
| `--context-length` | 512 | 最大 512 |
| `--stride` | 128 | 窗口起点间隔 |
| `--target-mode` | `all` | `all`/`last`/`tail` |
| `--target-length` | 空 | `tail` 时必填 |
| `--encode-batch-size` | 64 | Tokenizer 的批量，影响速度和显存，不影响语义 |
| `--samples-per-shard` | 4096 | 每个 `.pt` shard 的窗口数 |
| `--device` | `auto` | Tokenizer 运行设备 |

每个窗口由 `context_length+1` 根 K 线构造：前 `context_length` 根作为输入，向右移一位得到目标。归一化只使用输入历史窗口的统计量，不使用未来目标。

Cache 存储 S1/S2 `int16` token、时间 stamp、split/loss mask 及 manifest。Token 范围当前只有 0‑1023，因此 `int16` 与 `uint16` 同样每 token 2 字节，同时兼容旧 PyTorch 序列化器。

### 18.4 `all`/`tail`/`last` 的复用边界

- `all`：分割范围内的所有 next-token 位置参与 Loss；
- `tail N`：保留完整输入，仅最后 N 个目标位置参与 Loss；
- `last`：只最后一个目标位置参与 Loss。

已有 `all` Cache 可在加载时收窄为 `tail` 或 `last`，无需重新 Tokenize。这只是改变 `loss_mask`，不是在 512 根输入之外一次预测 N 根未来 K 线。反向无法复用，因为 `last`/`tail` Cache 没有保留更宽的有效监督 mask。

### 18.5 训练时的 shard 加载和 DDP 分片

Token shard 是物理存储单位，不是“提前绑定给某张 GPU”。训练时数据集懒加载 shard，不需要一次把全部 Token Cache 放入 RAM。接入 DDP 后，各 rank 通过分布式 sampler 获得不重复的窗口索引；模型梯度则在同步的 backward 中由 DDP/NCCL All-Reduce。

因此：

- 不需要离线生成 `gpu0`/`gpu1` 两份 Token Cache；
- 数据分片防止两卡重复训练同一窗口；
- All-Reduce 只同步梯度，不要求两卡读相同样本；
- `gradient_accumulation_steps>1` 时，中间 micro-batch 使用 `no_sync()` 本地累积，最后一个 micro-batch 才通信。

### 18.6 生成后的必查项

1. `bucket_manifest.json` 的源文件数、行数、symbol 数和 bucket 数合理；
2. Token Cache manifest 的 `context_length`、`stride`、`target_mode`、split 日期与训练命令一致；
3. Tokenizer fingerprint 与实验所用 Tokenizer 一致；
4. train/validation/test 都有样本，target count 不为 0；
5. 随机抽查一个窗口，输入、目标、stamp 和 mask 形状正确；
6. 正式训练先用相同 Cache 运行 smoke test，然后再提高 epoch 和 worker 数。

---

## 19. 从股票池到可比实验的标准流程

```text
固定股票池快照
  → plan 核对数量与 SHA-256
  → XtQuant download 或从全市场 filter-existing
  → validate Parquet
  → bucketize
  → tokenize 生成带 manifest 的 all Cache
  → smoke test/数据审查
  → 基座 teacher+predicted 评估
  → 单因子短实验
  → 保留统一 run 目录和评估 JSON
  → 仅对有收益方案扩大 epoch/数据/随机种子
```

每组对照实验至少锁定：股票池 SHA-256、源数据指纹、Tokenizer 指纹、split 日期、context/stride/target mask、optimizer step 预算、有效 batch、随机种子和评估 conditioning。任何一项不同，都不应把 Loss 差直接归因于 rank 或目标模块。

---

## 20. 当前服务器环境和已配好的资产

### 20.1 服务器基线

当前服务器已完成代码、Python 虚拟环境、预训练模型、数据和实验输出目录的配置，不需要为每次训练重建环境。统一工作目录为：

```text
/data/projects/Kronos
```

已知服务器硬件/软件记录：

| 项目 | 当前记录 |
|---|---|
| GPU | 2 张 NVIDIA Blackwell |
| NVIDIA 内核驱动 | 580.159.03，Open Kernel Module |
| Python | 3.12，虚拟环境路径显示为 `.venv/lib/python3.12` |
| PyTorch CUDA runtime | 已有运行记录为 13.0 |
| CUDA 可用性 | `torch.cuda.is_available()` 已返回 `True` |
| 单卡/双卡 | LoRA smoke test 都已通过 |
| 服务器 RAM | 约 377 GB |

上表是交接时的已知记录。每次重装驱动、PyTorch 或迁移机器后必须重新执行本节后面的环境核验，不要仅依赖文档记录。

### 20.2 已配好的路径

| 资产 | 服务器路径 | 状态/用途 |
|---|---|---|
| 项目源码 | `/data/projects/Kronos` | 已部署，所有命令从此目录执行 |
| Python 环境 | `/data/projects/Kronos/.venv` | 已安装并通过 smoke test |
| Kronos 基座 | `/data/projects/Kronos/pretrained/Kronos-base` | 已下载 |
| Kronos-small | `/data/projects/Kronos/pretrained/Kronos-small` | 已下载 |
| Tokenizer | `/data/projects/Kronos/pretrained/Kronos-Tokenizer-base` | 已下载 |
| A 股全市场 5m 原始数据 | `/data/projects/Kronos/data/lora` | 已有约一年 Parquet |
| 全市场 Token Cache | `/data/projects/Kronos/data/lora_token_cache_v1` | 已用于全市场训练/评估 |
| SW2 半导体 Token Cache | `/data/sectors/sw2_semiconductor/token_cache` | 已分桶、分片并用于板块实验 |
| 训练输出 | `/data/projects/Kronos/checkpoints` | 保存 run、best/last、resume 和 plots |

现有数据和 checkpoint 是服务器实验资产。维护目录时必须避免对 `.venv`、`pretrained`、`data` 和 `checkpoints` 执行递归删除、覆盖或带 `--delete` 的同步操作。

### 20.3 A 股一年 5 分钟 K 线数据审阅结果

数据下载说明已经在本文第 17 节给出，完整实现在：

```text
data_download/xtquant/cli.py
data_download/xtquant/downloader.py
data_download/xtquant/manifest.py
data_download/xtquant/validator.py
data_download/xtquant/README.md
```

数据来源是 XtQuant/MiniQMT，原始输出目录是：

```text
/data/projects/Kronos/data/lora/
├── download_manifest.json
├── universe.json
├── quality_report.json
├── failed_symbols.json
├── year=2025/month=MM/*.parquet
└── year=2026/month=MM/*.parquet
```

对当前交接数据中 manifest 和质量报告的审阅结果为：

| 字段 | 值 |
|---|---:|
| 下载状态 | `completed` |
| 周期 | `5m` |
| 日期范围 | 2025-09-01 至 2026-09-01 |
| 复权方式 | `front` |
| 股票池数量 | 5,591 |
| Parquet 文件数 | 728 |
| 总行数 | 63,572,025 |
| 完成 batch | 56 |
| 空数据 symbol | 30 |
| 失败 symbol | 0 |
| 包含北交所 | 是 |
| 包含下载当日 | 否 |
| 股票池 SHA-256 | `1cd19911e207f8d7530b247ac0c22333c3873486306eaace9857cf7aafb8ee99` |

因此，交接状态可以明确写为：“A 股全市场约一年 5 分钟 K 线已下载并物化为 Parquet，已有可复用 Token Cache，训练环境已配好。”30 个空 symbol 不等于下载失败，可能来自上市时间、退市或指定区间内无有效行情；其明细应以 `empty_symbols` 和 manifest 为准。

在服务器上可以用下列多行命令再次校验数据：

```bash
cd /data/projects/Kronos

./.venv/bin/python -m data_download.xtquant.cli validate \
  --output-dir /data/projects/Kronos/data/lora \
  --log-level INFO
```

参数说明：

- `cd /data/projects/Kronos`：切换到项目根目录，保证 Python 可以找到项目包；
- `./.venv/bin/python`：显式使用项目内已配好的虚拟环境；
- `-m data_download.xtquant.cli validate`：以模块方式运行统一数据校验入口；
- `--output-dir`：待检查的原始 Parquet 数据根目录；
- `--log-level INFO`：输出普通进度和质量信息。

`validate` 本身不需要连接 MiniQMT。只有重新执行 `plan`、`download`、`retry-failed` 或板块查询/导出时才需要可正常导入的 XtQuant 和已登录 MiniQMT。服务器已有完整数据时，普通训练和评估不需要 XtQuant。

---

## 21. 实验环境启动、核验和重建

### 21.1 已配好环境的日常启动

正常情况只需要激活环境，不要每次重新安装：

```bash
cd /data/projects/Kronos

source .venv/bin/activate

python -m pip check
```

参数/命令说明：

- `source .venv/bin/activate`：将当前 Shell 的 `python` 和 `pip` 切换到项目虚拟环境；
- `python -m pip check`：检查已安装包是否有缺失或版本冲突，不会改动环境。

检查 PyTorch、CUDA 和两张 GPU：

```bash
cd /data/projects/Kronos

./.venv/bin/python - <<'PY'
import torch

print("torch=", torch.__version__)
print("cuda_runtime=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("gpu_count=", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    print(
        index,
        torch.cuda.get_device_name(index),
        torch.cuda.get_device_capability(index),
    )
PY
```

`<<'PY' ... PY` 把多行 Python 代码通过标准输入交给项目虚拟环境，不会在项目中创建脚本文件。

- `torch.__version__`：当前 PyTorch wheel 版本；
- `torch.version.cuda`：该 wheel 编译时使用的 CUDA runtime，不是单纯的驱动版本；
- `torch.cuda.is_available()`：PyTorch 能否正常初始化 CUDA；
- `torch.cuda.device_count()`：当前进程可见 GPU 数，本服务器预期是 2；
- `get_device_capability`：核对 Blackwell compute capability。

如果 `nvidia-smi` 和 PyTorch 的 CUDA 初始化报 `Driver/library version mismatch`，这不是 LoRA 或 DDP 代码问题。在未重启时只能尝试使用与当前内核模块一致的库环境；若 NVML/内核模块已产生版本分裂，通常仍需要管理员重载驱动或重启。

### 21.2 新机器或环境损坏时重建

`requirements-train-server.txt` 故意不包含 PyTorch，因为 Blackwell 所需 wheel 必须与当时服务器驱动/CUDA 兼容性单独选择。仅在需要重建时执行：

```bash
cd /data/projects/Kronos

python3.12 -m venv .venv

source .venv/bin/activate

python -m pip install \
  --upgrade \
  pip \
  setuptools \
  wheel
```

随后先安装与服务器兼容的 PyTorch。当前 requirements 里的 cu130 命令是示例，必须以实际已验证 wheel 为准：

```bash
python -m pip install \
  torch==2.12.0 \
  --index-url https://download.pytorch.org/whl/cu130

python -m pip install \
  --requirement requirements-train-server.txt
```

参数说明：

- `python3.12 -m venv .venv`：在项目内新建 Python 3.12 隔离环境；
- `--upgrade`：升级基础包管理工具；
- `torch==2.12.0`：文档中保留的版本示例，不代表可在任意时间盲目重装；
- `--index-url .../cu130`：从 PyTorch CUDA 13.0 wheel 索引安装；
- `--requirement requirements-train-server.txt`：安装数据、Kronos runtime、Transformers、Accelerate 和 PEFT 的最小服务器依赖。

当前服务器环境已可用，所以上述重建命令不是下一次训练的必须步骤。在覆盖 `.venv` 之前，应先记录已工作环境：

```bash
cd /data/projects/Kronos

mkdir -p \
  logs

./.venv/bin/python -m pip freeze \
  > logs/environment-freeze-before-rebuild.txt
```

该 freeze 文件用于恢复或对照服务器环境，不是训练输入。

### 21.3 环境重建后的 LoRA smoke test

```bash
cd /data/projects/Kronos

CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -m lora.smoke_test \
  --base-model-path /data/projects/Kronos/pretrained/Kronos-base \
  --tokenizer-path /data/projects/Kronos/pretrained/Kronos-Tokenizer-base \
  --adapter-output-dir /data/projects/Kronos/checkpoints/Kronos-base-QV-LoRA \
  --batch-size 1 \
  --sequence-length 512 \
  --lora-target-profile self-attn-qv-s1 \
  --lora-r 8 \
  --lora-alpha 16 \
  --device cuda:0
```

| 参数 | 作用 |
|---|---|
| `CUDA_VISIBLE_DEVICES=0` | 让进程只看到物理 GPU 0 |
| `--base-model-path` | 本地 Kronos Predictor 基座目录 |
| `--tokenizer-path` | 本地冻结 Tokenizer 目录 |
| `--adapter-output-dir` | 用于模型配置的默认 adapter 根目录；未传 `--save-adapter` 时 smoke test 不会保存模型 |
| `--batch-size 1` | 合成输入的 batch 大小 |
| `--sequence-length 512` | 合成 token 序列长度，用最大训练长度验证显存和形状 |
| `--lora-target-profile` | 此次要构建和验证的 LoRA 模块组合 |
| `--lora-r` / `--lora-alpha` | LoRA 秩和缩放参数 |
| `--device cuda:0` | 将 Tokenizer 和 Predictor 放在当前可见的第 0 张 GPU |

smoke test 应输出命中模块、可训练参数数量、S1/S2 logits 形状和 `LoRA smoke test passed.`。

---

## 22. 服务器正式训练命令格式与参数解释

### 22.1 命令行格式审阅

本文第 17、18、20‑22 节中的可执行 Linux 命令都使用多行 Bash 格式：

- 首行写入口；
- 每个重要参数独立一行；
- 未结束行以反斜杠 `\` 续行；
- 最后一行不加反斜杠；
- 不要把文档中的 `<run-name>` 等占位符原样粘贴执行；
- Linux Bash 使用 `\`，Windows PowerShell 使用反引号，两者不要混用。

这种格式比单行命令更容易确认 cache、stride、target mask、rank 和输出目录，也更容易做 A/B 对照。

### 22.2 单卡训练模板

以已配好的 A 股全市场 Token Cache 为例：

```bash
cd /data/projects/Kronos

CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u -m finetune_lora.train \
  --finetune-mode lora \
  --base-model-path /data/projects/Kronos/pretrained/Kronos-base \
  --tokenizer-path /data/projects/Kronos/pretrained/Kronos-Tokenizer-base \
  --adapter-output-dir /data/projects/Kronos/checkpoints/Kronos-base-QV-LoRA \
  --token-cache-path /data/projects/Kronos/data/lora_token_cache_v1 \
  --context-length 512 \
  --stride 128 \
  --target-mode all \
  --epochs 6 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --warmup-steps 500 \
  --lora-target-profile self-attn-qv-s1 \
  --lora-r 8 \
  --lora-alpha 16 \
  --s1-loss-weight 0.5 \
  --s2-loss-weight 0.5 \
  --training-s2-conditioning teacher \
  --num-workers 16 \
  --validation-interval 6000 \
  --checkpoint-interval 2000 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.001 \
  --gradient-log-interval 50 \
  --precision bf16 \
  --device cuda:0 \
  --run-name a5m_base_joint_all_c512s128_saqvs1_r8a16_tch_eb16_lr1e-4_wu500_e6_sd100
```

| 参数组 | 作用 |
|---|---|
| `CUDA_VISIBLE_DEVICES=0` + `--device cuda:0` | 单卡运行并绑定唯一可见 GPU |
| `-u` | Python 标准输出不缓冲，日志能及时写出 |
| `-m finetune_lora.train` | 运行 LoRA/full 统一训练入口 |
| `--finetune-mode lora` | 冻结基座，只训练 PEFT adapter 和指定的可保存模块 |
| 三个 model/output path | 指定 Predictor、Tokenizer 和 run 输出根目录 |
| `--token-cache-path` | 直接读已预处理 shard，训练期间不再运行 Tokenizer |
| `context/stride/target` | 必须与 Token Cache manifest 一致；它们决定窗口和 Loss mask |
| `--epochs 6` | 最多训练 6 轮，可能因 epoch-level early stopping 提前结束 |
| `batch-size=4` + `accumulation=4` | 单卡有效 batch 为 16 个窗口/每 optimizer step |
| `learning-rate` + `warmup-steps` | 前 500 个 optimizer step 线性升温，之后按 cosine schedule 变化 |
| profile/rank/alpha | 决定 LoRA 命中模块、参数量和 `alpha/r` 缩放 |
| S1/S2 weight | 联合 Loss 中两项的相对权重 |
| `teacher` | 训练 S2 时使用真实 S1 token 作为条件 |
| `--num-workers 16` | DataLoader worker 数，需监控 RAM、CPU 和 IO，不是 GPU 数 |
| `--validation-interval 6000` | 每 6000 个 optimizer step 执行一次完整验证，epoch-end 仍会验证 |
| `--checkpoint-interval 2000` | 每 2000 optimizer step 保存可恢复 checkpoint |
| early-stopping 两参数 | epoch-end Loss 至少下降 0.001 才重置 3 轮 patience；best model 仍跟踪任何严格更低 Loss |
| `--gradient-log-interval 50` | 每 50 optimizer step 记录分组梯度范数 |
| `--precision bf16` | 启用 Blackwell 支持的 BF16 训练 |
| `--run-name` | 唯一 run 目录名；已存在同名时不覆盖 |

该命令是格式模板，不代表下一轮科学实验必须使用这组超参数。执行前必须用 cache manifest 确认 stride 真的是 128，并更换为未使用的 run name。

### 22.3 双卡 DDP 命令模板

```bash
cd /data/projects/Kronos

CUDA_VISIBLE_DEVICES=0,1 ./.venv/bin/torchrun \
  --standalone \
  --nproc-per-node=2 \
  -m finetune_lora.train \
  --finetune-mode lora \
  --base-model-path /data/projects/Kronos/pretrained/Kronos-base \
  --tokenizer-path /data/projects/Kronos/pretrained/Kronos-Tokenizer-base \
  --adapter-output-dir /data/projects/Kronos/checkpoints/Kronos-base-QV-LoRA \
  --token-cache-path /data/projects/Kronos/data/lora_token_cache_v1 \
  --context-length 512 \
  --stride 128 \
  --target-mode all \
  --epochs 6 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-4 \
  --warmup-steps 500 \
  --lora-target-profile self-attn-qv-s1 \
  --lora-r 8 \
  --lora-alpha 16 \
  --s1-loss-weight 0.5 \
  --s2-loss-weight 0.5 \
  --training-s2-conditioning teacher \
  --num-workers 4 \
  --validation-interval 6000 \
  --checkpoint-interval 2000 \
  --precision bf16 \
  --run-name a5m_base_joint_all_c512s128_saqvs1_r8a16_tch_eb16_lr1e-4_wu500_e6_ddp2_sd100
```

DDP 额外参数说明：

- `CUDA_VISIBLE_DEVICES=0,1`：让 `torchrun` 看到两张 GPU；
- `./.venv/bin/torchrun --standalone`：显式使用项目虚拟环境内的 launcher，并在单机内自动建立 rendezvous；
- `--nproc-per-node=2`：启动两个训练进程，每个进程绑定一张 GPU；
- 不再传 `--device cuda:0`：训练代码使用 `LOCAL_RANK` 自动绑定每个进程；
- 此例有效 batch = 每卡 4 × 累积 2 × 2 卡 = 16；
- DDP 只支持 Token Cache 模式，不能直接在多进程中使用原 Parquet 在线 Tokenize；
- 训练窗口由 sampler 按 rank 分片，梯度同步由 DDP/NCCL 自动完成；
- 日志和模型文件只由 rank 0 写入。

---

## 23. 实验环境和训练前检查表

1. 工作目录仍为 `/data/projects/Kronos`；
2. `.venv`、`pretrained`、`data`、`checkpoints` 未被删除或覆盖；
3. `pip check` 通过，PyTorch 能看到 2 张 Blackwell GPU；
4. 单卡 LoRA smoke test 通过；
5. 当前 cache manifest 与训练命令的 context/stride/target/split/Tokenizer 指纹相同；
6. 使用新 run name，不覆盖原实验；
7. 先运行一个短周期实验，确认数据、Loss、验证、checkpoint 和 plot 都正常产生；
8. 正式实验保存 `config.json`、`dataset_summary.json`、`environment.json` 和环境 freeze 文件。

---

## 24. 训练参数、模型权重和实验结果保存位置

### 24.1 运行开始时如何确定保存目录

训练启动后会在终端打印：

```text
run_directory=<resolved-run-directory>
```

实际 run 目录按下式构造：

```text
<adapter-output-dir>/runs/<run-name>/
```

服务器上当前 Kronos-base LoRA 的标准位置是：

```text
/data/projects/Kronos/checkpoints/Kronos-base-QV-LoRA/runs/<run-name>/
```

其中：

- `--adapter-output-dir` 可自定义最外层实验根目录；
- `--run-name` 可自定义实验子目录名；
- 未传 `--run-name` 时，LoRA 训练自动生成 `YYYYMMDD_HHMMSS_r<rank>_s<stride>`，全参数训练生成 `YYYYMMDD_HHMMSS_full_s<stride>`；
- 自动名称发生冲突时增加 `_01`、`_02` 等后缀；
- 显式 `--run-name` 已经存在时会拒绝覆盖；
- `--model-display-name` 可自定义前端展示名，但不改变磁盘 run 路径。

`config.json`、`best_model`、`last_model`、`resume`、`metrics.jsonl`、`result.json` 和 `plots` 等 run 内部名称当前没有 CLI 开关可自定义。这些名称同时被恢复、评估和前端发现代码使用，不应在训练后手工改名。

例如：

```bash
cd /data/projects/Kronos

CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u -m finetune_lora.train \
  --adapter-output-dir /data/projects/Kronos/checkpoints/Kronos-base-QV-LoRA \
  --run-name sw2semi_base_s1_tail280_c512s128_saqvs1_r8a16_e20_sd100 \
  --model-display-name "SW2半导体 S1 LoRA · Kronos-base" \
  '<other-training-arguments>'
```

`<other-training-arguments>` 是说明用占位符，不能原样执行。完整训练命令见第 22 节。

### 24.2 “训练参数”分别保存在哪里

“参数”可能指实验超参数、学习到的模型权重或 optimizer 恢复状态，三者保存位置不同：

| 内容 | 保存位置 | 用途 |
|---|---|---|
| 命令行/数据/训练/LoRA 超参数 | `config.json` | 复现实验，包括 base、cache、rank、alpha、Loss 权重、batch、LR 等 |
| 最优验证权重 | `best_model/` | 推理和正式评估的默认选择 |
| 训练结束时权重 | `last_model/` | 用于分析过拟合、与 best 对比 |
| 按 step 恢复权重 | `resume/step-XXXXXXXX/` | 中断后继续训练 |
| Optimizer 状态 | `resume/step-XXXXXXXX/optimizer.pt` | 恢复 AdamW 动量和参数组 |
| Scheduler 状态 | `resume/step-XXXXXXXX/scheduler.pt` | 恢复 warmup/cosine 所在位置 |
| epoch/step/早停等 | `resume/step-XXXXXXXX/trainer_state.json` | 恢复训练计数器和签名校验 |
| Python/NumPy/PyTorch/CUDA RNG | `resume/step-XXXXXXXX/rng_state_rank-XXXX.pt` | 让每个 DDP rank 尽可能精确续训 |
| 最新可恢复 checkpoint 指针 | `resume/latest.json` | `--resume-from <run-dir>` 时解析最新 step |

LoRA 和全参数模式的权重文件不同：

#### LoRA 模式

```text
best_model/
├── adapter_config.json
├── adapter_model.safetensors
├── deployment.json
└── README.md                  # 由 PEFT save_pretrained 产生时可能存在
```

`adapter_model.safetensors` 只保存 LoRA B/A 矩阵和 `modules_to_save`（例如完整解冻的 `proj_s1`），不包含整个 Kronos 基座。`adapter_config.json` 的 `base_model_name_or_path` 和 `deployment.json` 记录所需基座，加载时仍需要对应的 `/data/projects/Kronos/pretrained/Kronos-*`。

#### 全参数模式

```text
best_model/
├── config.json
├── model.safetensors
└── deployment.json
```

`model.safetensors` 包含完整 Predictor 权重，因此体积显著大于 LoRA adapter。`last_model/` 使用与当前微调模式相同的文件结构。

### 24.3 best、last 和 resume 的区别

- `best_model/`：任何 step-level 或 epoch-end 验证的 selection Loss 严格低于历史最佳时立即更新；
- selection Loss 由当次的 `s1-loss-weight`、`s2-loss-weight`、`training-s2-conditioning` 及混合分支配置决定，当前不支持任意指定“按 S1 top-1 最高”作为 best 规则；
- `last_model/`：早停或所有 epoch 结束后的最后权重；
- `resume/step-*`：不是用于部署的最佳模型，而是包含 optimizer/scheduler/RNG 的完整训练快照；
- `--checkpoint-interval N` 可自定义每 N 个 optimizer step 保存一次 resume，N 必须大于 0；
- `--validation-interval N` 可自定义 step-level 验证频率，设为 0 会关闭 step-level 验证，但 epoch-end 验证仍保留。

中断训练使用：

```bash
cd /data/projects/Kronos

CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u -m finetune_lora.train \
  --resume-from /data/projects/Kronos/checkpoints/Kronos-base-QV-LoRA/runs/<run-name> \
  '<same-training-arguments>'
```

`--resume-from` 会恢复原 run，不创建新 run；已经生成 `result.json` 的完整 run 不能原地 resume。如果只想从原 adapter 权重开始一次新实验，使用 `--initialize-from-adapter`，它会新建 optimizer、scheduler、RNG 和 run name。

### 24.4 每个 run 会产生哪些结果

```text
<run-dir>/
├── config.json
├── dataset_summary.json
├── environment.json
├── metrics.jsonl
├── history.json
├── result.json
├── best_model/
├── last_model/
├── resume/
└── plots/
    ├── loss.svg
    ├── loss.png
    ├── component_loss.png
    ├── conditioning_loss.png          # 有条件分支指标时
    ├── token_accuracy.png
    ├── token_entropy.png
    ├── learning_rate.png
    ├── gradient_norms.png             # 开启梯度日志且有数据时
    └── gradient_conflict.png          # 开启冲突诊断且有数据时
```

| 文件 | 包含内容 |
|---|---|
| `config.json` | 完整数据、训练和 LoRA/full 配置，解析后的 run name 与路径 |
| `dataset_summary.json` | train/validation/test 窗口数、时间边界、Token Cache 摘要和数据指纹 |
| `environment.json` | Python/PyTorch/CUDA/GPU/DDP world size 等环境信息 |
| `metrics.jsonl` | 最完整的时序记录：train、step/epoch validation、test、梯度和梯度冲突，每行一个 JSON |
| `history.json` | epoch-end 训练/验证摘要，适合按 epoch 对比 |
| `result.json` | 完整训练结束后的最终摘要和 best-model test 结果 |
| `plots/` | Loss、S1/S2 分项、token top-k、熵、学习率和可选梯度诊断曲线 |
| `best_model/` / `last_model/` | 可加载权重、PEFT/模型配置和前端 `deployment.json` |

`metrics.jsonl` 中可包含：

- train/validation/test 的总 Loss、S1 Loss、S2 Loss 和 learning rate；
- S1/S2 top-1、top-5 accuracy 以及 mean entropy；
- teacher/predicted/mixed S2 Loss、teacher S2 top-1/top-5/熵及 predicted-minus-teacher gap；
- `grad_norm_total`、`grad_norm_proj_s1`、`grad_norm_transformer`、`grad_norm_dependency`；
- S1/S2 共享 Transformer 梯度范数、比值和余弦相似度。

`result.json` 只在训练正常完成、保存 last model，并用 best model 完成 test 评估后生成。若任务中断，没有 `result.json` 并不代表之前什么都没保存；应检查 `metrics.jsonl`、`history.json`、`best_model/`、`plots/` 和 `resume/latest.json`。

终端 stdout/stderr 当前不会自动成为 run 内文件；可复现的数值以 `metrics.jsonl` 和 JSON 结果为准。如果还需要保留完整终端文本，应在启动命令外层使用 `tee` 写入带 run name 的日志，并避免多次实验共用同一日志文件。

### 24.5 独立评估结果保存位置可自定义

`evaluate_base` 和 `evaluate_adapter` 不会强制把 JSON 写入 run 目录。使用 `--output-json` 可自定义完整输出文件路径；如果不传，结果只打印到终端。

推荐保存在对应 run 的 `evaluations/` 子目录：

```bash
cd /data/projects/Kronos

mkdir -p \
  /data/projects/Kronos/checkpoints/Kronos-base-QV-LoRA/runs/<run-name>/evaluations

CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u -m finetune_lora.evaluate_adapter \
  --base-model-path /data/projects/Kronos/pretrained/Kronos-base \
  --tokenizer-path /data/projects/Kronos/pretrained/Kronos-Tokenizer-base \
  --adapter-path /data/projects/Kronos/checkpoints/Kronos-base-QV-LoRA/runs/<run-name> \
  --model-selection best \
  --token-cache-path /data/projects/Kronos/data/lora_token_cache_v1 \
  --split validation \
  --s2-conditioning all \
  --context-length 512 \
  --stride 128 \
  --target-mode all \
  --batch-size 4 \
  --num-workers 16 \
  --precision bf16 \
  --device cuda:0 \
  --output-json /data/projects/Kronos/checkpoints/Kronos-base-QV-LoRA/runs/<run-name>/evaluations/eval_validation_all_best.json
```

重要参数：

- `--adapter-path`：可传 run 目录，也可直接传 `best_model/` 或 `last_model/`；
- `--model-selection best|last`：传 run 目录时选择哪一组权重；
- `--split validation|test`：选择评估数据分割；
- `--s2-conditioning all`：同时运行 teacher、greedy、sample、top-k、shuffle、random 和 constant 等条件/消融路径；
- `--output-json`：可自定义的评估结果路径，不要使用“最新.json”这类会覆盖历史的名字。

评估结果文件应保留 adapter/base 路径、split、conditioning、window/target count、Loss、top-k accuracy、entropy 和条件 gap。

### 24.6 未在本文正文逐一列出的实验到哪里查看

权威来源始终是服务器上的完整 run：

```text
/data/projects/Kronos/checkpoints/<experiment-root>/runs/<run-name>/
```

可以用 `rg --files` 查找所有训练摘要和评估 JSON：

```bash
cd /data/projects/Kronos

rg --files \
  /data/projects/Kronos/checkpoints \
  | rg \
    '(^|/)(config|dataset_summary|environment|history|result)\.json$|metrics\.jsonl$|/evaluations/.*\.json$'
```

查看某个 run 的完整超参数和最终摘要：

```bash
cd /data/projects/Kronos

./.venv/bin/python -m json.tool \
  /data/projects/Kronos/checkpoints/<experiment-root>/runs/<run-name>/config.json

./.venv/bin/python -m json.tool \
  /data/projects/Kronos/checkpoints/<experiment-root>/runs/<run-name>/result.json

tail \
  --lines 20 \
  /data/projects/Kronos/checkpoints/<experiment-root>/runs/<run-name>/metrics.jsonl
```

查看图表文件：

```bash
ls \
  -lh \
  /data/projects/Kronos/checkpoints/<experiment-root>/runs/<run-name>/plots
```

### 24.7 快速定位规则

```text
想复现实验       → config.json + dataset_summary.json + environment.json
想看每个step的变化 → metrics.jsonl
想按epoch对比      → history.json
想看最终测试摘要   → result.json
想做推理/部署      → best_model/ + 匹配基座
想比较过拟合       → best_model/ vs last_model/
想中断续训         → resume/latest.json 或 resume/step-*/
想看曲线           → plots/
想看条件消融       → evaluations/*.json
想审阅未列实验     → 查服务器 checkpoints/*/runs/*
```
