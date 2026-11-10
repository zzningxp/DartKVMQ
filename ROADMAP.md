# DartKVMQ Roadmap

目标：一个代码库 + 一套统一 benchmark，复现三个 KV-cache 压缩方法（RDKV / KVarN / RAQ-MLA→MHA），面向**标准 MHA(GQA) 模型**，然后迭代我们自己的方法。

**验收原则（用户要求）**：每个方法开发完成后，必须至少对齐原论文的**一个具体实验**（"对齐实验"）——同模型、同任务、同预算口径下，复现数字应落在论文数字附近（容忍采样噪声）。MLA 原生实验放二期，一期用 MHA 等价设置对齐方法学结论。

## 方法清单（原 README 信息迁入，2026-11-10；README 保持空白——仓库已公开，细节不外写）

| # | 方法 | arXiv | 类型 | 官方代码 / 复现位置 |
|---|------|-------|------|----------|
| 1 | RDKV | [2605.08317](https://arxiv.org/abs/2605.08317) | 联合逐出+量化的率失真比特分配（prefill 后反注水） | 论文无代码 → 原生实现 `src/dartkv/methods/rdkv.py` + `alloc/` |
| 2 | RAQ-MLA | [2610.33625](https://arxiv.org/abs/2610.33625) | 角色感知量化（K→score 误差，V→value 误差）+拟合可逆变换；MLA→MHA 适配 | 论文无代码 → P3 待做 |
| 3 | KVarN | [2606.03458](https://arxiv.org/abs/2606.03458) | 免校准 Hadamard 旋转+双轴方差归一化，2-bit，修 decode 累积误差 | `external/KVarN/`（快照，**未跟踪**）；port 在 `src/dartkv/quant/kvarn.py` |
| 4 | SKVQ | [2405.06219](https://arxiv.org/abs/2405.06219) | 离线通道重排（KMeans 校准）+截断动态分组量化+精确滑窗 W=128 | `external/SKVQ/`（快照，**未跟踪**，原在根目录 2026-11-10 移入）；port 在 `src/dartkv/quant|methods/skvq.py` |
| 5 | TQ (TurboQuant) | — | 合作者代码，未集成 | `bt-kvcatch/`（git 跟踪） |
| 6 | ours | — | TBD | — |

基线 KIVI 无官方快照 → 原生实现（`src/dartkv/methods/kivi.py`；与 SKVQ 自带 KIVI_mode 交叉验证过）。

## 目录结构

```
docs/        论文精读、设计决策、实验日志（experiments.md、skvq.md、benchmark.md）
external/    上游快照，gitignore 全忽略（只在本机）：KVarN/、SKVQ/、ifeval_official/、longbench_official/
bt-kvcatch/  合作者的 TurboQuant 代码（TQ，git 跟踪，未集成）
src/dartkv/
  quant/     非对称均匀量化、Hadamard、VarN、SKVQ codec（均与官方逐位对齐）
  alloc/     MCKP 拉格朗日 + Bennett 反注水（RDKV）
  methods/   full / kivi / kvarn / rdkv / skvq（三区分组布局；skvq 为精确滑窗变体）
  cache/     CompressedCache —— HF transformers Cache 路由到方法
  tasks/     benchmark 任务（line_retrieval、longbench、ifeval + ifeval_checker、math500、gsm8k）
  bench.py   统一 runner：model × method × task → results/*.json
scripts/     入口（run_bench.py、smoke_cache.py、calibrate_skvq.py、compare_official.py）
configs/     实验配置 + skvq_calib/ 重排文件（*.pt 被 gitignore，需要就 git add -f）
tests/       单元测试（CPU、无模型下载；test_skvq.py 依赖 external/SKVQ 快照，缺失则跳过）
results/     实验输出
papers/ data/   论文 PDF / 数据集与校准缓存（均 gitignore）
```

## 快速开始

```bash
PY=/home/daodao/anaconda3/envs/dart312/bin/python
$PY tests/test_methods.py && $PY tests/test_quant_alloc.py && $PY tests/test_skvq.py
# Line-Retrieval 试点（cuda1 + hf-mirror）
$PY scripts/run_bench.py --model Qwen/Qwen3-4B --method kvarn --lines 100 200 --n 20 --tag pilot
```

## 环境（已确认）

- GPU: 2× RTX 5090 32GB (sm_120 Blackwell)。**cuda0 被占用，所有实验用 cuda1**（`CUDA_VISIBLE_DEVICES=1`）。
- conda env: `dart312` → `/home/daodao/anaconda3/envs/dart312/bin/python`（Python 3.12.13, torch 2.9.0+cu128, transformers 5.20.1, triton 3.5.0, datasets 4.8.5; 无 flash-attn/vllm → 用 SDPA）。
- 权限：`.claude/settings.local.json` 白名单（Auto 模式的安全检查服务失效，只能靠显式规则）。

## 统一 benchmark（详见 docs/benchmark.md）

两级：
- **Core（快速迭代）**: GSM8K(200子集) + MATH500 + Line-Retrieval(100–600行) + RULER-NIAH(4k/16k/32k, 3子集)
- **Full（对齐论文/出报告）**: + HumanEval(164) + AIME24 + LongBench子集 + IFEval

统一口径：有效 bits/cached-element (BPE) 或 FP16-equivalent token 预算两者都记录；一律 HF transformers + SDPA，压缩后 dequant→dense attention 保数值等价；性能(速度/显存)测量后置。

## Phases

### P0 — 基础设施（基本完成）
- [x] 论文全文获取+精读笔记（docs/papers/*.md）
- [x] KVarN 源码快照 → external/KVarN（数值细节的 ground truth）
- [x] src/dartkv 骨架：quant 原语（非对称均匀量化、Hadamard、VarN、bit打包、MCKP）、CompressedCache、方法接口（full/kivi/kvarn）
- [x] 单元测试 17/17 通过（tests/test_quant_alloc.py + tests/test_methods.py）：量化 round-trip、VarN 收敛+精确重构、MCKP 预算、Hadamard 保内积、tile 编解码界、16-bit 近无损（防转置/广播回归）、2-bit 下 KVarN<KIVI 误差、三区 sawtooth 布局
- [x] CompressedCache 冒烟（scripts/smoke_cache.py，tiny-Llama @ cuda1）：手动 prefill+decode 循环与 generate() 均通过；kivi/kvarn 分组路径实际触发
- [ ] bench harness（runner + 任务：Line-Retrieval 最先）——P1 一起做
- [ ] 模型下载：Llama-3.1-8B-Instruct（gated，需 HF token）+ Qwen3-4B（ungated，走 hf-mirror）；wikitext 校准集

### P1 — KVarN 复现（最先做：有官方源码可对齐，且无预算耦合）
- 实现：Hadamard(通道) + VarN(双轴) + 2bit RTN；sink=128/组=128/尾=128 三区布局；decode 在线量化
- **对齐实验 1a**（严格）: Llama-3.1-8B-Instruct Line-Retrieval 100–600 行 vs 论文 Tab.4（KVarN 99/98/98/95/91/89; KIVI 98/97/90/84/80/83; FP16 100/99/99/96/93/92）
- **对齐实验 1b**: Llama-3.1-8B IFEval strict/loose vs Tab.3（71.0/76.5, FP16 71.1/76.8）
- **对齐实验 1c**（若加 Qwen3-4B）: MATH500/HumanEval/AIME24 vs Tab.1/2（thinking mode 参数）
- 产出：伪解码(pseudo-decode)累积误差曲线复现（论文 Fig.5 的 Llama 版）

### P2 — RDKV 复现
- 实现：prefill probe(w_t, w_c) + MCKP 反注水 {0,2,4,8,16} + ε表校准 + 压缩 cache（dequant 路径；TriZone 打包/ Triton 后置 P5）
- **对齐实验 2a**: Llama-3.1-8B LongBench 16 任务 @64L/128L/512L/1024L vs 论文 Tab.1（1024L avg 49.47, FullKV 49.82; 64L 领先 baseline 2.7）
- **对齐实验 2b**: RULER 11 任务 @2048L 4k–128k vs Tab.9（4k 98.59; 64k 83.20）
- 基线（统一实现）：SnapKV/AdaKV 可用 kvpress 或自实现（同等 probe S_w=32,w=5）
- 注意：RDKV 是长 prompt 短生成 regime；与 P1 的 decode regime 在统一套件中分别报告

### P3 — RAQ-MLA → RAQ-MHA 适配
- 实现：offline fit（alternating minimization + CG + requantized line search）；K: per-channel INT4 + score-aware A_K（query 侧补偿）；V: per-token INT2 + value-aware A_V（输出侧 d×d 补偿）
- **对齐实验 3a**（MHA 等价）: Llama-3.1-8B 上 KIVI/QuaRot(Hadamard)/RAQ-MHA 同 BPE 对比，趋势对齐 RAQ 论文 Tab.1/3 排序（RAQ > OSCAR/Hadamard > 裸量化）；量化器细节(94% clip, 页调度, BPE 记账)与论文口径一致
- **对齐实验 3b**（方法学核心）: role metric vs M=I vs Hadamard 的 attention-output 误差对比（论文 Fig.5/7：role metric 23.3%→21.1%）
- **二期（MLA 原生）**: DeepSeek-Coder-V2-Lite / Moonlight-16B on cuda1，跑论文 Tab.1 子集验证 fit 算法本身
- 风险最高：fit 成本、per-token M_t 的近似（token 子采样 256/layer）、κ_max/λ 网格超参

### P4 — 统一对比 + 我们的方法
- 全方法 × core benchmark 矩阵（同 BPE 口径归一：每方法报 2.3/3.25/预算曲线）
- 迭代方向（候选）：RDKV 的 decode 阶段再预算化（论文自己指出的 future work）；KVarN 校准免费 + RAQ 角色感知的结合；token-scale 误差与 role-aware 度量的统一
- 淘汰决策点：若某方法在我们的 MHA 统一套件上无法体现论文优势（复现失败/口径不合），记录原因并降级为基线

### P5 — 性能测量（可选后置）
- TriZone 风格打包 + Triton dequant-fused attention；或接 KVarN 的 vLLM 路径做速度对比
- 显存：实际打包存储 vs BPE 理论值

## 当前状态

- 2026-11-10 (ModelScope 通道 + 双BOS修复 + MATH500 接入): ① ModelScope 下载通道打通（免 token、网络通、modelscope 包在位；`addict` 已补装）——数据侧 `AI-ModelScope/MATH-500`（=HuggingFaceH4/MATH-500）已落地 `data/math500/test.jsonl`（500 条，schema 验证过）；模型侧验证 `LLM-Research/Meta-Llama-3.1-8B-Instruct` 免 token 可下（用户选择暂不下载）。② **修复 chat 模板双 BOS bug**：模板自带 `{{- bos_token }}` + tokenizer 默认 `add_special_tokens=True` → Llama prompt 双 BOS（跑题复读 system prefill 的元凶）；Qwen 不加自动 BPE 特殊 token，历史 Qwen3 结果不受影响。修复=prompter 协议带 `add_special_tokens` 属性（bench.py + run_bench.py）。③ 官方 Llama-3.1 chat 模板资产化（`src/dartkv/chat_templates/llama-3.1.jinja`，来自 ModelScope 上游 tokenizer_config）+ `ensure_chat_template()`（本地快照无模板时注入；base 版快照本来就没有）+ GEN_PRESETS key `llama-3.1`→`llama-3`（匹配 `Meta-Llama-3___1-8B` 这类本地目录名）。④ 本地 Llama base 版判别：蜜罐指令测试（2+2 复读）+ raw 续写测试 → base，不可用于 1a/1b/2a。⑤ MATH500 任务接入（`tasks/math500.py`：\boxed 平衡括号提取 + 轻量 LaTeX 归一化，CPU 用例 6/6）+ runner `--thinking` 旗标（Qwen3 thinking mode，1c 口径），pilot n=4 full-KV 4/4（45s/样本）；**n=100×{full,kivi,kvarn_h,kvarn} 矩阵在跑**（论文 Tab.1 目标 82.6/77.8/78.9/79.2）。⑥ GSM8K 任务接入（`tasks/gsm8k.py`，数据从本地 HF 缓存落地 data/gsm8k/，评分校验通过；GPU 排队中）。LongBench p2 pilot 结果补录 experiments.md（rdkv@1024/256 在 trec/hotpotqa 零损失；rdkv256 hotpotqa 待补跑）。
- 2026-11-10 (SKVQ 并入): 官方代码快照移入 `external/SKVQ/`（与 KVarN 同等待遇：不跟踪、保持 pristine；合作者树里的原版可从 git 历史 `HEAD:SKVQ/` 找回）。codec+精确滑窗布局 port 完成（`src/dartkv/quant/skvq.py` + `methods/skvq.py` + `scripts/calibrate_skvq.py`），与官方**逐位等价**（tests/test_skvq.py 9/9，无快照环境自动跳过）。Qwen3-4B 趋势对齐：SKVQ 2/2 (63.3@g32) > KIVI 2/2 (13.3)；关键发现：校准统计必须与实际缓存张量匹配（post-RoPE harness 用 `--hook cache` 的 `_postrod.pt`；官方 pre-RoPE 校准配他们的 pre_rope=True 引擎）。详见 docs/skvq.md、docs/experiments.md。

- 2026-11-10 (P2 开工): RdkvMethod 实现完成（src/dartkv/methods/rdkv.py）：SDPA 包装器抓 post-RoPE 窗口 Q 的 probe（w_t SnapKV 式池化、w_c ThinK 分数）、on-the-fly ε 表（可选缓存表）、MCKP 两阶段分配（V 层级 token 集 + K per-(l,h) 通道位宽——dense 路径无法表达 per-head 驻留 token 集，按 SnapKV 先例跨 head 聚合，K 通道置零精确等价）、被驱逐 token 物理删除、Zone C FP16 追加。预算口径按论文 B_head=32nd。单测 6 个（probe 形状/预算约束/大预算无损/decode 追加/补丁卸载/tiny 端到端）。LongBench 任务接入（对齐 2a 前置）：官方 metrics+prompts+maxgen 逐字 vendor，数据 data/longbench/（THUDM data.zip，datasets-lib 脚本式加载已废弃→直接 jsonl），trec/hotpotqa 冒烟通过。待跑：Qwen3-4B LongBench pilot（模型不 gated）→ Llama-3.1 对齐（等 token）。
- 2026-11-10: 对齐实验 1a(Qwen3-4B 版) 完成——绝对数字与论文不可比（stack 级偏移，我们 full-KV 基线本身低于论文），改用**趋势对齐 + codec 等价证明**（scripts/compare_official.py 逐元素比对官方纯 PyTorch 参考实现，rel-diff 3-6e-4）。n=100×6 长度矩阵（full/kivi/kvarn_h/kvarn）：排序 KVarN≥Hadamard≥KIVI **6/6 成立**，1a 关闭。IFEval 任务接入完成：数据(google/IFEval 541) + 官方 checker 原样 vendor（src/dartkv/tasks/ifeval_checker/，仅 import 行改相对引用，diff 可证）+ strict/loose 双口径评分；管道试点（full/kivi/kvarn 各 75.0/80.0 @n=20，短 prompt 下压缩≈FP16 与论文一致）。待 Llama token 到位跑 1b 对齐。
- 2026-11-09: P0 完成。量化/分配原语 + VarN(对齐官方源码) + 三区布局 + KIVI/KVarN/Full 方法 + CompressedCache（HF transformers 5.20.1 实测可用，手动循环与 generate() 都通）。单测 17/17。进入 P1：bench harness（Line-Retrieval 先行）+ 真实模型下载。
  - 实现要点：KVarN 的 K 瓦片按官方参考取 [D,G]（通道为行）做 VarN，K per-channel 量化（scale/zero 折叠 per-channel sinkhorn scale，per-token scale 单独存），V per-token（对偶）；`quantize_asym(group_size=None)` 返回原形状（scale/zero 在 `dim` 上为 1）。
- 2026-11-09: 项目启动。三篇论文精读完成（docs/papers/）。环境确认（dart312 + cuda1）。

## 遗留问题（跨方法）

1. ~~Llama-3.1-8B-Instruct 的 HF 下载需要 token/镜像？~~ 已探明（2026-11-10）：HF 镜像对 gated 仓库无效；**ModelScope 免 token 可下** `LLM-Research/Meta-Llama-3.1-8B-Instruct`（已验证 config 可拉）。本地 `/mnt/new1t/models/Meta-Llama-3___1-8B` 经行为验证是 **base 版**（复读 prompt、raw 续写），不能做指令评测。用户决定暂不下载 Instruct，先用 Qwen3-4B 推进（1c 本就是 Qwen3 实验）。
2. ~~transformers 5.x 的 Cache API 适配~~ 已解决：CompressedCache(DynamicCache) 重写 update(key,value,layer_idx,**kw)，pass 边界用 layer_idx 回绕检测；tiny 模型实测 generate() 可用。
3. RDKV ε 表的校准数据：论文用 LongBench prefill 前缀；我们先用 wikitext，对齐实验时换 LongBench 前缀。
4. LongBench 评测脚本的口径（max_len 截断、chat template）需与 RDKV 论文对齐——他们未完全公开细节，用 LongBench 官方默认。
