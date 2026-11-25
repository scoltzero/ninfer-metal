# NAX–ANE crossover 解析模型 v0

> 日期：2026-09-19  
> 状态：**理论投影，不是 M5 Ultra 实测**  
> 目标：先确定解析结构、可证明的上界和最小标定集；M5 Ultra 到货后只替换硬件常数与效率函数，不重写 scheduler 判据。  
> 适用范围：GEMM / qmm 的 NAX-only 与 NAX+ANE 选择。Attention、GDN recurrence 和完整模型的端到端收益另按 Amdahl 比例汇总。

## Material Passport

- Artifact ID：`nax-ane-crossover-v0`
- 类型：解析性能模型与计算包络
- 输入：公开 M5/H17s 测量、Apple 公布的 Ultra 资源比例、仓库中的 Qwen3.8-27B oQ4e 真实形状
- 验证状态：`ANALYZED`；算术已由 `tools/nax_ane_crossover.py` 复算，硬件投影尚未在 M5 Ultra 验证
- 主要限制：Ultra 线性缩放、NAX/ANE 同时运行不降频、固定开销与形状无关、可独立寻址的 ANE 实例暂按 2 个处理，均只是 v0 假设

## 1. 先区分三个问题

单设备选择是：

\[
T_N(x)=T_A(x)
\]

混合路径启用边界是：

\[
T_N(x)=\min_{\alpha,a}T_H(x,\alpha,a)
\]

其中 \(a\) 是切分轴。完整模型的端到端收益还要再乘上“可由混合 GEMM 加速的时间占比”，不能把单 GEMM 的上界直接当成整模型收益。

按当前投影，NAX 的大 GEMM 计算顶约为 ANE 单 matmul 的 6.06 倍，聚合内存带宽也更高，且 ANE 还有 layout/dispatch 成本。因此，只要 \(\eta_N\) 没有在某个小形状上严重坍塌，**单设备边界很可能是空集：大部分支持域都由 NAX-only 支配**。ANE 的主要价值不是替代 NAX，而是与 NAX 并行增加总吞吐。

## 2. v0 计算包络

按给定的同代资源比例做第一阶投影：

\[
P_{N,0}=30.9\times4=123.6\ \mathrm{TFLOP/s}
\]

\[
P_{A,\mathrm{single}}=10.2\times2=20.4\ \mathrm{TFLOP/s}
\]

\[
P_{A,\mathrm{fused\ slope}}=19.6\times2=39.2\ \mathrm{TFLOP/s}
\]

这三个数是“由 M5/H17s 测量按资源数线性外推”的投影，不是 M5 Ultra 实测。尤其 `19.6` 是融合链的边际 slope，不是任意单 GEMM 都能获得的吞吐。

若 NAX 与 ANE 并行时完全不互相降速，计算受限下：

\[
\alpha^*=\frac{P_A}{P_N+P_A}
\]

| 模式 | ANE 最优工作份额 | 理想总吞吐 | 单 GEMM 理论上限 |
|---|---:|---:|---:|
| ANE 单 matmul ceiling | 14.17% | 144.0 TFLOP/s | 1.165×（+16.50%） |
| fused-slope 上界 | 24.08% | 162.8 TFLOP/s | 1.317×（+31.72%） |

第二行只能作为融合子图的上界。它要求融合真正减少 dispatch/中间回写，并且 live activation 没有把 ANE 推出片上工作集；不能拿它替代单 GEMM 的 `20.4 TFLOP/s`。

## 3. 并发降速后的正确闭式解

令：

- \(Q_{N0}\)：NAX-only 的有效吞吐；
- \(Q_{NH}\)：NAX 与 ANE 同时运行时 NAX 的有效吞吐；
- \(Q_{AH}\)：同时运行时 ANE 的有效吞吐。

计算受限时最优切分仍可解析：

\[
\alpha^*=\frac{Q_{AH}}{Q_{NH}+Q_{AH}}
\]

但 hybrid 能赢的必要条件变成：

\[
Q_{NH}+Q_{AH}>Q_{N0}
\]

若满足该条件，固定增量开销为 \(\Delta L\) 时：

\[
\boxed{
F^*=\frac{\Delta L}
{1/Q_{N0}-1/(Q_{NH}+Q_{AH})}
}
\]

当 \(Q_{NH}=Q_{N0}\) 时，它退化为：

\[
F^*=\Delta L\frac{P_N(P_N+P_A)}{P_A}
\]

这说明还必须标定一个原列表中没有单列的常数：**NAX 与 ANE 同时运行时的并发降速/争用系数**。如果 ANE 启用使 NAX 降速超过 ANE 补回的吞吐，则不存在 crossover，ANE 永远不应启用。

## 4. 4096×4096 示例复算

对 \([M,4096]\times[4096,4096]\)：

\[
F/M=2\times4096^2=33.554432\ \mathrm{MFLOP}
\]

使用 `123.6 + 20.4 TFLOP/s` 的单 GEMM 包络：

| Hybrid 增量开销 | 计算包络的 \(M^*\) |
|---:|---:|
| 50 μs | 1,300 |
| 100 μs | 2,600 |
| 110 μs | 2,860 |
| 150 μs | 3,900 |
| 200 μs | 5,200 |

所以原先“50/100/150/200 μs 对应约 1.3K/2.6K/3.9K/5.2K”是正确的。110 μs 是公开 M5/H17s **ANE 内部成本模型**的 dispatch floor；它不是完整 hybrid 的 \(\Delta L\)。后者还要包含 input pack、GPU/ANE fence、输出 merge、tail、可能的 layout 转换，因此只能先把 110 μs 当作乐观锚点。

## 5. Qwen3.8-27B 真实形状的计算包络

下表使用仓库 `bench/probes/qwen38_27b_oq4e_shapes.json` 的真实 \(K,N\)，取 \(\Delta L=110\ \mu s\)。同一 \(KN\) 的 gate/up 与 down 有相同的纯计算交点，但内存、片上工作集和切分可行性并不相同。

| 形状 | K×N | 单 matmul 包络 \(M^*\) | fused-slope 上界 \(M^*\) |
|---|---:|---:|---:|
| MLP gate/up 或 down | 5,120×17,408 | 538 | 317 |
| GDN qkv | 5,120×10,240 | 915 | 538 |
| Attention q | 5,120×12,288 | 763 | 449 |
| GDN z / out | 5,120×6,144 | 1,525 | 897 |
| Attention k/v | 5,120×1,024 | 9,153 | 5,385 |
| lm_head | 5,120×248,320 | 37.7 | 22.2 |
| GDN a/b | 5,120×48 | 195,255 | 114,878 |

这些值只说明“若两侧都在计算 roof 上，多少行足以摊薄 110 μs”。它们不是 scheduler 阈值：lm_head 的输出流量、ANE 支持上限、INT8 procedure 体积和数值误差都可能让理论交点不可实现；a/b 则显然不值得单独交给 ANE。

## 6. 4.72 MB 折点的正确含义

公开 reverse-engineering 资料中的 4.72 MB 是 M5 ANE 的**片上 live activation working-set threshold**，不是整份权重大小。超过它以后，中间 activation 需要出片回写，融合链会从高 slope 掉到 streaming regime。

对于按 N 切、\(n_A\) 个可独立寻址的 ANE 实例平均承担总 ANE 输出份额，最乐观的每实例 live bytes 至少为：

\[
W_i\gtrsim M\left(b_AK+b_C\frac{\alpha N}{n_A}\right)
\]

因此：

\[
M_{\rm SRAM}\lesssim
\frac{4.72\times10^6}
{b_AK+b_C\alpha N/n_A}
\]

公开的 4.72 MB 是单个 M5/H17s 测量值；不能因为 M5 Ultra 有 32-core Neural Engine 就直接把它当成一个可聚合的 9.44 MB SRAM。下面暂按 \(n_A=2\) 计算，这只是由总 core 数比例和现有 oMLX 双实例路径得到的工程投影，实际实例拓扑必须在目标机 probe。

以 BF16/FP16 activation 和 output 计，单-matmul \(\alpha=14.17\%\)、暂取 \(n_A=2\)：

- `K=4096,N=4096`：只计 input 时约 576 行；计 input+本实例 output 时约 538 行；
- `K=5120,N=17408` gate/up：约 461 / 372 行；
- `K=17408,N=5120` down：约 136 / 133 行。

这还是忽略 scratch 和融合中间量的乐观上限。由此可见：4096 方阵在 `M≈2860` 的 crossover 已远过 4.72 MB 折点，不能继续使用 fused-slope 上界；MLP gate/up 的 fused 上界 `M≈317` 恰好靠近片上边界，更需要实机确认，而 down 投影会更早越界。

## 7. Roofline 与 hybrid 的完整形式

对设备 \(d\) 的某个实际切片 \(x_d\)：

\[
Q_d(x_d)=P_d\eta_d(x_d)
\]

\[
T_d(x_d)=L_d+
\max\left(
\frac{F_d}{Q_d},
\frac{D_{d,\mathrm{weight}}}{B_{d,\mathrm{weight}}},
\frac{D_{d,\mathrm{activation}}}{B_{d,\mathrm{activation}}}
\right)
+T_{d,\mathrm{layout}}+T_{d,\mathrm{epilogue}}
\]

Hybrid 不能简单把两个独立 roofline 相加。一个可标定的第一版是：

\[
T_H=L_{\rm prepare}+L_{\rm sync}+T_{\rm merge}
+\max\left(
T_{N,\mathrm{work}},
T_{A,\mathrm{work}},
\frac{D_{\rm shared}}{B_{\rm UMA,H}}
\right)
\]

其中 \(B_{\rm UMA,H}\) 和 \(Q_{NH},Q_{AH}\) 必须在同时运行时测量。单独测出的 `P_N`、`P_A`、`B_N`、`B_A` 不能证明它们可加。

NAX 的几何效率也应补上 K 维：

\[
\eta_{\rm tile}\approx
\frac{MKN}
{t_M\lceil M/t_M\rceil\,
 t_N\lceil N/t_N\rceil\,
 t_K\lceil K/t_K\rceil}
\]

当前真实形状的 K/N 多数已按 32/64 对齐，但切分后的 \(\alpha N\) 还必须满足 NAX `BN=64`、ANE procedure 输出粒度和双 ANE 均衡，因此 \(\alpha\) 是离散集合，不是任意实数。

## 8. 切分轴结论需要加“内存域价格”

在两边都从同一层级冷读数据、每字节代价相同的简化条件下：

\[
D_{\rm dup,M}=b_WKN,\qquad
D_{\rm dup,N}=b_AMK
\]

所以 \(N\)-split 更好的条件确实是：

\[
\frac{M}{N}<\frac{b_W}{b_A}
\]

但当前 oMLX 路径把选中的权重预先重量化成 per-output-channel INT8 并放入固定 ANE procedure bank，GPU suffix 继续使用 q4/q5；权重和 activation 不处在相同的内存域。更一般的条件应写成：

\[
c_A b_AMK<c_W b_WKN
\]

其中 \(c_A,c_W\) 是由驻留、缓存命中、实际带宽和并发争用决定的每字节时间价格。当前实现还只支持固定形状的 N-split，因此 v0 scheduler 的 `axis` 实际上是受约束的可行集，而不是 M/N 两者任意选。

## 9. 总上下文长度不是每次 GEMM 的 M

在连续、未分块的 prefill 中可以写 \(M=B\times S\)。但现有 oMLX 会按固定 chunk 执行，ANE program 也是固定序列长度（当前常见为 2048）。因此真实变量是：

\[
M=B\times S_{\rm chunk}
\]

若总上下文为：

\[
S_{\rm total}=J S_{\rm chunk}+r
\]

则总时间是：

\[
T(S_{\rm total})=
J\,T(S_{\rm chunk})+T(r)+L_{\rm request}
\]

长上下文带来的收益来自“更多完整、ANE-eligible 的 chunk”和一次性成本摊薄，而不是把单次 GEMM 的 M 自动变成 8K/32K。最终 scheduler surface 至少还需要 `chunk_size` 与 `tail_policy` 两个变量。

## 10. 从单 GEMM 上界到完整模型

如果 NAX-only 时间中只有比例 \(f\) 属于可加速 GEMM，单 GEMM 加速为 \(s\)，端到端上界为：

\[
S_{\rm e2e}\le
\frac{1}{(1-f)+f/s}
\]

例如 \(f=70\%\)：

- 单 matmul 包络 `s=1.165` 时，端到端上限约 `1.109×`（+10.9%）；
- fused-slope 上界 `s=1.317` 时，端到端上限约 `1.203×`（+20.3%）。

所以 `+16.5%` 与 `+31.8%` 是 eligible GEMM 区段的理想上界，不是整个 Qwen prefill 的直接承诺。

## 11. 需要标定的最小参数集

原先 10–20 个微基准的思路仍成立，但参数集应补齐为：

1. NAX solo：\(P_N,B_N,L_N,\eta_N(M,K,N,q)\)；
2. ANE solo：单 matmul与融合子图分别的 \(P_A,B_{A,w},B_{A,a},L_A,C_{SRAM}\)；
3. concurrent：\(Q_{NH}/Q_{N0}\)、\(Q_{AH}/Q_{A0}\)、\(B_{UMA,H}\)，至少在低/中/高三档 split 测量；
4. bridge：input pack、fence、merge 的固定项和按字节 slope；
5. Ultra 拓扑：可独立寻址的 ANE 实例数、实例间不均衡、四 die/UltraFusion 访问 penalty；
6. dtype：q4/q5 NAX 解量化，ANE INT8 重量化后的速度、驻留体积与质量门；
7. chunk/tail：固定 2048 program、多 chunk 与 residual tail 的成本。

这些仍可由少量正交微基准拟合，不需要扫完整五维表。拟合后再定义：

\[
G(x)=T_N(x)-\min_{\alpha\in\mathcal A,\,a\in\mathcal F}T_H(x,\alpha,a)
\]

其中 \(\mathcal A\) 是满足 tile/alignment 的离散 split 集，\(\mathcal F\) 是实现实际支持的切分轴集合。最后才对分段的 \(G(x)=0\) 做二分或离散搜索。

## 12. 当前结论

没有数学上的卡点。现在已经能得到：

- 正确的 compute-only closed form；
- 并发降速后仍成立的 generalized closed form；
- 真实 Qwen GEMM 形状的条件化 crossover；
- 4.72 MB activation 折点与 chunking 对边界的约束；
- “单设备边界很可能为空、真正重要的是 hybrid 边界”的可检验判断。

尚不能提前给出的只有一个东西：**M5 Ultra 上不带条件的最终 token 阈值**。缺的不是数学，而是 simultaneous NAX+ANE 的有效吞吐、完整 \(\Delta L\)、固定 chunk/tail 行为和 ANE INT8 近似路径的实测常数。

## 13. 证据来源

- Apple Newsroom，M5 Ultra：80-core GPU、32-core Neural Engine、1.2 TB/s UMA、四 die UltraFusion：<https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/>
- Bryngelson, *Apple Neural Engine: Architecture, Programming, and Performance*，M5/H17s 的 9.5–10.2 TFLOP/s 单 matmul、19.6 TFLOP/s fused slope、145 GB/s weight stream、4.72 MB activation threshold、0.11 ms cost-model floor：<https://arxiv.org/abs/2606.22283>
- 当前 oMLX ANE/GPU prefill 的固定 shape、N-split、INT8 procedure bank、dual-instance 与 chunk/tail 行为：`omlx/docs/experimental/qwen35_ane_prefill.md`
- Qwen3.8-27B oQ4e 的实际 K/N/bit 形状：`bench/probes/qwen38_27b_oq4e_shapes.json` 与 `docs/quantization.md`
