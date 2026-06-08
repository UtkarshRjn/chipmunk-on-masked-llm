# Mask-Adaptive Sparse Kernels for Masked Diffusion Language Models

**Training-free Hopper-native acceleration via Chipmunk-style gather-pack kernels, magnitude-based K/V delta selection, and composition with existing block caching and speculative decoding.**

Author: Utkarsh Ranjan
Target advisor: Dan Y. Fu (UCSD; co-author of Chipmunk)
Target venues: ICLR 2027 workshops (SLLM, DeLTa) as primary near-term target; MLSys / ICLR / NeurIPS main tracks as longer-term aspirational targets
Status: Draft v2 — corrected against primary sources, May 2026

---

## 0. One-line pitch

The two most relevant prior works on accelerating masked diffusion language models — FastDLLM (Wu et al. 2025) and dKV-Cache (Ma et al. 2025) — both operate at the algorithm/operator level and explicitly flag system-level/hardware-aware work as future. Chipmunk's kernel toolkit (`cp.async` scattered gather, SRAM packing with `wgmma`, top-k delta selection, voxel reordering) is exactly the missing piece. This proposal ports that toolkit to MDM inference, replaces dKV-Cache's recency heuristic with magnitude-based active selection, and stacks composably with FastDLLM and DualDiffusion. Expected: 1.3–2× on top of FastDLLM specifically in compute-bound regimes (HumanEval, large batch, very long context) where existing block caching gets only 3–4×.

---

## 1. Executive summary

**The problem.** Masked Diffusion Language Models (MDMs) like LLaDA-8B (Nie et al. 2025), Mercury (Inception Labs), and Dream-7B do not support traditional KV cache because their attention is fully bidirectional — every token's K/V can shift between denoising steps. This forces O(L²) attention recompute every step. For an 8B model at 1024 generation length and 50–100 steps, this is 10–30× slower than a comparable AR LM with KV cache.

**Existing acceleration is algorithmic.** Two recent works approximate the missing cache:
- **FastDLLM** (Wu et al. 2025): block-wise approximate KV cache + confidence-aware parallel decoding. Achieves up to 27.6× on long-generation memory-bound benchmarks (GSM8K gen-1024) but only 3.7–4× on compute-bound code generation (HumanEval).
- **dKV-Cache** (Ma et al. 2025): caches K/V one step after a token transitions from masked to decoded; periodic refresh. Achieves 2–10× across benchmarks. They implement an operator-level `concat_reorder` to gather scattered cached tokens contiguously — exactly the structure Chipmunk uses, but at PyTorch op level not kernel level.

**The gap, in the authors' own words** (dKV-Cache §5 verbatim):

> "One primary limitation of this work lies in its focus on algorithmic design in isolation. While our proposed method introduces an effective caching mechanism from a purely algorithmic perspective, diffusion language models also exhibit substantial room for improvement at the system level. We believe that future research integrating algorithmic innovations with system-level optimization, such as memory management, parallelism, and hardware-aware execution, could unlock further efficiency gains and performance improvements for DLMs."

**The opening.** Chipmunk (Silveria, Govande, Fu 2025) developed exactly this system-level toolkit for image/video DiTs: `cp.async`-based scattered gather from HBM, contiguous SRAM packing, dense `wgmma` GEMM on packed tile, magnitude-based top-k active selection, periodic dense re-anchor, voxel-style token reordering for column-aligned chunked sparsity, all benchmarked on H100. **The Chipmunk paper flags AR LMs as a possible extension but does not mention MDMs.** Neither FastDLLM nor dKV-Cache does any kernel-level work; both ran on A100/A6000/H20, not H100.

**This proposal.** Port the Chipmunk machinery to MDM inference. Replace dKV-Cache's recency-based active set (cache after decoding + delay) with magnitude-based top-k (positions whose K/V drifted most). Implement on Hopper with `cp.async` + warp specialization in the ThunderKittens DSL. Stack with FastDLLM as block-level coarse caching. Compose with DualDiffusion (Goyal et al. 2026) as a verifier acceleration in speculative decoding. Provide a theoretical error bound that DualDiffusion's heuristic verification punted on.

**Realistic targets:**
- **Per-step kernel speedup**: 2.5–4× over dense attention at MDM-typical sparsity ratios (translating Chipmunk's 9.3× attention-only number to MDM-shape sparsity).
- **End-to-end speedup over FastDLLM**: 1.3–2× in compute-bound regimes (HumanEval, batch >1, long context). Lower (1.0–1.2×) in heavily memory-bound regimes where FastDLLM already saturates.
- **Stacked end-to-end vs vanilla LLaDA-8B**: 30–50× on long-generation tasks; 8–15× on compute-bound tasks.

**Why this is the right project for Dan's group.** Dan Fu is the senior author on Chipmunk. ThunderKittens (Spector, Fu et al. 2024) is the kernel DSL. The MDM literature is fresh (LLaDA, FastDLLM, dKV-Cache, DualDiffusion all 2025–2026) and entirely unclaimed at the kernel level. The methodology transfers nearly mechanically. Cross-pollination paper at the Hazy / Sandy Research / MDM-LM intersection.

---

## 2. Background and prior work landscape

### 2.1 Masked Diffusion Language Models — what they are and why caching is hard

MDMs reverse a forward masking process. For sequence x of length L, the forward process at time t ∈ [0,1] independently masks each token with probability t. At t=1, all tokens are `[MASK]`; at t=0, the original sequence. The reverse process (LLaDA, MDLM, SEDD, D3PM lineage) trains a transformer mask predictor `p_θ(x | x_t)` to predict all currently-masked tokens simultaneously.

**Inference loop:**
```
1. Initialize x_1 = [MASK]^L
2. For t = T..1:
     - Forward pass: mask predictor predicts all masked positions
     - Choose subset to unmask (typically by confidence or random)
     - x_{t-1} updates with newly unmasked tokens
3. Return x_0
```

For LLaDA-8B with L=1024 and T=128 steps, this is 128 forward passes, each with full bidirectional attention — O(T·L²) total complexity. AR with KV cache: O(T·L) per token × L tokens = O(L²). MDMs do roughly **T = 64–128 forward passes** vs AR's **L = 1024** sequential token passes, but each MDM pass is more expensive because no caching.

**Why bidirectional attention breaks caching** (formalized in dKV-Cache §3.2):

1. **Timestep-variant K/V**: In AR with causal mask, K[1..t-1] and V[1..t-1] never change. In MDM, every token attends to every position; when any position unmasks, every K[i] can shift. Formally, `K_t^[i] ≠ K_{t'}^[i]` for `t ≠ t'`.

2. **Non-sequential decoding order**: AR generates strictly L→R. MDM unmasks any position based on per-step confidence, so we can't pre-determine which positions to compute Q/K/V for at step t.

**The empirical out** (dKV-Cache §3.1, Fig 2):
> "Once a token is decoded, its representation becomes relatively stable in subsequent steps, whereas the representations of still-masked tokens continue to fluctuate significantly."

This is the empirical foundation any caching-based MDM acceleration leans on. In principle K/V can change every step; in practice, the bulk of representations stabilize after their token is decoded. The unanswered question is *which* representations are still actively shifting on a given step. Recency (dKV-Cache's heuristic) is one signal; magnitude (Chipmunk's) is another.

### 2.2 Existing acceleration methods — corrected positioning

| Method | Approach | Mechanism | Speedup achieved | Hardware tested |
|---|---|---|---|---|
| **FastDLLM** (Wu et al. 2025) | Block-wise approximate KV cache; confidence-aware parallel decoding; theorem-bounded parallel decoding | PyTorch ops; standard attention kernels | **Up to 27.6×** on GSM8K-1024 8-shot; **3.7–4×** on HumanEval | A100 80GB |
| **dKV-Cache-Decode** (Ma et al. 2025) | One-step delayed caching after token decoded; periodic refresh; `concat_reorder` op for gather/scatter | PyTorch ops; explicitly *not* kernel-level | 2–10× across benchmarks; up to 7× on long prefill | A6000, H20 |
| **dKV-Cache-Greedy** (Ma et al. 2025) | Tighter cache lifespan with local window; O(L²) total complexity | PyTorch ops | Higher speedup, lower accuracy | Same |
| **DualDiffusion** (Goyal et al. 2026) | Algorithmic spec decoding; FastDLLM as drafter, LLaDA as verifier; heuristic remasking (KL-div, confidence) | No kernel work; composes existing models | 3.9× on MMLU; **fails on GSM8K** (0.25 vs 0.57) | Unspecified GPU |
| **Esoteric LMs** (Sahoo, Yang et al. 2025) | Pure causal attention denoiser; trained from scratch with hybrid AR+MDM objective | Architectural change; full retraining required | 14–65× vs MDMs; 3–4× vs BD3-LMs | Academic-scale (LM1B, OWT) only |
| **Block Diffusion / BD3-LMs** (Arriola et al. 2025) | Semi-autoregressive: block-diffusion within blocks, AR across blocks; partial KV cache | Architectural change; new training | Speedup vs MDM but quality drops at low NFE | Academic-scale |

**Three crucial framing points:**

1. **No method does kernel-level work on Hopper.** FastDLLM ran on A100 (no `cp.async`, no TMA stack, no warp specialization). dKV-Cache ran on A6000 and H20. The Hopper-specific feature surface — `cp.async` for scattered loads, TMA for contiguous loads, `wgmma` for tensor-core GEMM, warp-spec for producer/consumer overlap — is **completely uncharacterized for MDM workloads**.

2. **The closest method (dKV-Cache) explicitly invites the kernel work.** Quoted at length above. They have the algorithmic structure but call out hardware-aware execution as future work.

3. **Eso-LMs is a different niche.** It's training-from-scratch (~9K H200 hours just for ablations) at academic scale. There's no 8B Eso-LM checkpoint in the wild. For accelerating *existing* released models (LLaDA-8B, Mercury-Coder, Dream-7B) — which is what real users care about — Eso-LMs doesn't compete.

### 2.3 Chipmunk's machinery — what's available to port

Chipmunk (Silveria, Govande, Fu 2025) accelerates image/video DiT inference. Its core insight: across denoising steps, only 5–25% of intermediate activation values explain 70–90% of cross-step change. The other 70–95% can be cached. Recompute only the active subset.

The shared computational form they exploit:
```
   attention output  =  softmax(Q @ K^T) @ V                  =  act(a @ b) @ c
   MLP output        =  GeLU(x @ W_1) @ W_2                   =  act(a @ b) @ c
```

Both are weighted sums of vectors in `c`, where the weights come from `act(a @ b)`. The cross-step delta is:
```
   o_t  ≈  o_cache  −  (old contributions from active set)  +  (new contributions)
```
Active set selection uses top-k by column-sum magnitude (attention) or token-merged tile-mean delta (MLP).

**The kernel toolkit they built:**

1. **`cp.async`-based scattered gather**: Each thread issues independent loads from arbitrary HBM addresses to a contiguous SRAM tile. K parallel transactions, each ~64B; saturates HBM at ~85–90% of dense bandwidth.

2. **SRAM packing with dense `wgmma`**: Once the scattered K/V columns land in SRAM as a contiguous tile, run a normal tensor-core GEMM. The tensor cores never know the upstream gather was sparse.

3. **Voxel reordering for column sparsity**: Reorder tokens once at the start so each chunk of **C=192** tokens corresponds to a 3D spatial voxel. Chunk shares one sparsity pattern. Reduces approximation error 2× vs unstructured sparsity at same speed (Chipmunk Table 2).

4. **Periodic dense re-anchor**: Every M sparse steps (default 10), a full dense step refreshes the cache and identifies new sparsity patterns. Prevents error accumulation.

5. **Fused kernel for top-k + cache update**: Sparsity pattern identification overlapped with attention compute via warp specialization. 3.42× speedup vs PyTorch top-k for the pattern selection alone.

**Reported numbers on H100-SXM5:**
- **9.3×** attention kernel vs FlashAttention-3 at 93% sparsity
- **2.16×** end-to-end on HunyuanVideo (1030s → 477s)
- **3.72×** stacked with TeaCache step caching on HunyuanVideo
- **1.41×** end-to-end on FLUX.1-dev; 2.25× stacked

**Crucially, Chipmunk operates on output deltas in image diffusion, not K/V deltas in language MDMs.** The structural analogy is direct (scattered active set → gather → pack → dense GEMM), but the algorithmic substrate is different. Adapting requires:
- Defining what "active" means for K/V in MDMs (recency, magnitude, hybrid)
- Handling RoPE position embeddings under permutation
- Reasoning about bidirectional cascade error (image diffusion errors are visually imperceptible; text errors can change tokens and meaning)

---

## 3. The proposal: kernel-level Chipmunk for MDMs

### 3.1 Core thesis

> Across MDM denoising steps, the active set of token positions whose K/V representations have shifted meaningfully is small (consistent with dKV-Cache's empirical observation that representations stabilize post-decoding). This active set is *scattered* across the sequence in HBM. Existing methods cache K/V coarsely (FastDLLM block-wise) or by recency (dKV-Cache after-decode delay). Both leave throughput on the table because:
> 1. Coarse blocks don't isolate the actually-changing positions
> 2. Recency is a weak proxy for actual representation drift
> 3. Neither uses Hopper's gather/pack/wgmma machinery to make the dynamic selection cheap
>
> A magnitude-based active selection, gathered via `cp.async` and packed for dense `wgmma`, with periodic dense re-anchor and voxel-aligned token reordering, recovers Chipmunk's hardware efficiency in the MDM setting. Composes with FastDLLM (block-level) and DualDiffusion (spec-decoding level) for stacked speedup.

### 3.2 What gets contributed

Five distinct pieces, in order of decreasing certainty:

#### Contribution 1 — Hopper kernel for mask-adaptive attention

A `cp.async` + warp-specialized attention kernel where Q is computed only at **active** positions (still-masked + recently-changed) and K/V are split into two partitions:

- **Cached partition**: K/V for stable (long-decoded) positions in HBM, organized in column-major chunks of 192.
- **Active partition**: K/V for the active set, gathered via scattered `cp.async` and packed contiguously into SRAM.

Combined attention via FlashAttention-3-style two-pass online softmax across both partitions. Output written back to HBM with overlapped TMA store.

```cuda
// Pseudo-kernel structure
// Input: query_active_indices, kv_active_indices, kv_cache_blocks
// Output: attn_out for active query positions

// Stage 1 — Pack active queries (top-half of warps)
for k in 0..num_active_q:
    cp.async(Q_smem[k], Q_hbm + q_active_indices[k] * row_stride);

// Stage 2 — Pack active K/V (bottom-half of warps, parallel with Stage 1)
for k in 0..num_active_kv:
    cp.async(K_smem[k], K_hbm + kv_active_indices[k] * row_stride);
    cp.async(V_smem[k], V_hbm + kv_active_indices[k] * row_stride);

cp.async.wait_group(0);

// Stage 3 — Two-pass online softmax across cached + active
// Pass 1: compute attention against active partition (dense wgmma)
attn_active = wgmma(Q_smem, K_smem, V_smem);

// Pass 2: compute attention against cached partition (TMA loads, dense wgmma)
attn_cached = wgmma_with_tma(Q_smem, K_cache_blocks, V_cache_blocks);

// Stage 4 — Combine via FlashAttention online softmax reduction
output = combine_softmax(attn_active, attn_cached);

// Stage 5 — Async writeback
cp.async.bulk_store(out_hbm + q_active_indices, output);
```

Implemented in ThunderKittens DSL. Reference target: 60–75% of FlashAttention-3 dense throughput at 70% active-set sparsity (gives ~2–3× per-step speedup vs full dense recompute).

Three things make this nontrivial:
1. **Combined softmax across partitions.** Standard FlashAttention pattern but two simultaneous K/V sources. Two-pass online softmax with partial sums.
2. **Position embedding handling.** RoPE bakes position into Q/K. Two options: apply RoPE *after* gather (carrying position indices alongside), or pre-rotate in HBM. Default to apply-after since dKV-Cache showed this is feasible.
3. **Cache structure in HBM.** Need fast block reads (for cached partition) and fast scattered writes (when active positions commit back to cache after re-anchor). Use Chipmunk's column-major chunked layout; benchmark vs row-major.

#### Contribution 2 — Magnitude-based active subset selection

Replace dKV-Cache's **recency-based** active set (cache K/V one step after decoding, refresh every N steps) with a **hybrid magnitude + recency** signal:

```
   Active set A_t = A_recent ∪ A_magnitude
   
   A_recent     = {positions still masked} ∪ {decoded in last R steps}
   A_magnitude  = top-k by ||K_t^pred - K_{t-1}||  using a cheap proxy
                  (e.g., norm of single-layer feature, or early-exit head)
   
   |A_t| ≈ 25-40% of L
```

The recency component bounds the worst case (newly-masked or newly-decoded positions are guaranteed-active). The magnitude component adapts to actual drift — captures cases where an old token's representation still shifts due to bidirectional context updates from elsewhere.

Why this is a genuine algorithmic contribution: dKV-Cache's Figure 2(c) shows that the largest changes in K/V occur "exactly at the decoding step" — they observed the magnitude signal but used it only as motivation, not as a selection criterion. Chipmunk uses magnitude-based top-k for output deltas in images. Nobody has used it for K/V deltas in MDMs.

#### Contribution 3 — Periodic dense re-anchor with formal error bound

Every M denoising steps, run a full O(L²) bidirectional attention that:
- Recomputes K/V for all positions
- Resets cumulated error
- Refreshes the cached partition's contents

Hyperparameter sweep: M ∈ {3, 5, 7, 10, 15, 20}. Chipmunk uses M=10 for image diffusion; expected sweet spot for MDM is M ∈ [5, 10] given dKV-Cache's empirical refresh schedule (4–8) for similar reasons.

**The theoretical contribution**: an error bound under bidirectional cascade. Let:
- `err_step` = max ‖K_t^pred − K_t^true‖ across positions for one sparse step
- `τ` = magnitude threshold for active set inclusion  
- `M` = anchor period
- `κ_softmax` = numerical precision factor for online softmax combine

**Claim (informal):**
```
   err_step  ≤  C₁ · τ + κ_softmax · ε_numeric
   err_M     ≤  M · err_step  +  amplification term from bidirectional cascade
   err_after_anchor  =  0
```

The amplification term is the load-bearing piece of the analysis. For images, errors are visually imperceptible and bound trivially. For text, an error in K at position i can shift attention output at position j, which feeds back at next step, etc. The bound needs to capture this feedback. Likely tractable via a Lipschitz-style argument over attention dynamics, with the cascade growth factor depending on attention sharpness.

**Even a loose bound is a contribution.** DualDiffusion's verification rules (KL-divergence threshold, confidence threshold) are heuristic with no analysis; this would be the first formal error bound in the MDM acceleration literature. Sets M and τ from a target token-error rate rather than empirical sweeping.

#### Contribution 4 — Hopper-native characterization

Run the full method on H100-SXM5 with all Hopper-specific features:
- `cp.async` for scattered loads
- TMA (Tensor Memory Accelerator) for cached-partition loads (contiguous chunks)
- `wgmma` warp-group matrix multiply
- Producer/consumer warp specialization
- Async barrier-based softmax reduction

Compare to:
- Same method on A100 (no `cp.async` group-async, no TMA, no `wgmma`)
- dKV-Cache on H100 (just running their PyTorch code on better hardware)
- FastDLLM on H100 (same)

This benchmark alone — running the existing methods on Hopper — is a useful contribution, since neither author tested on Hopper.

#### Contribution 5 — Composition with FastDLLM and DualDiffusion

Stack three layers of acceleration:

```
   Layer 1 — Block-level: FastDLLM's confidence-aware parallel decoding +
             approximate KV cache. Coarse but high-impact in memory-bound regime.

   Layer 2 — Position-level: This proposal's mask-adaptive kernel. Within each
             block, recompute only the active subset's K/V. Compounds with
             Layer 1 by giving FastDLLM cheaper per-step compute.

   Layer 3 — Step-level: DualDiffusion-style speculative decoding. Use a small
             distilled drafter with aggressive Layer 1+2 settings; verify with
             larger model using conservative settings. Layer 2 makes the
             verifier itself ~2× faster.
```

The composability story matters because it sidesteps "do we beat FastDLLM" entirely. We *include* FastDLLM as a component.

### 3.3 What the algorithm looks like end to end

```python
# Pseudo-Python at the algorithm level

class MaskAdaptiveMDMInference:
    def __init__(self, model, M_anchor=7, sparsity=0.7, voxel_size=192):
        self.model = model
        self.M = M_anchor
        self.k = int(model.seq_len * (1 - sparsity))
        self.kv_cache = HBMCache(layout='column_major_192')
        self.step_count = 0
    
    def step(self, x_t, mask_state):
        if self.step_count % self.M == 0:
            # Dense re-anchor step
            K_full, V_full = self.model.compute_kv_dense(x_t)
            self.kv_cache.refresh(K_full, V_full)
            attn_out = self.model.dense_attention(x_t, K_full, V_full)
            # Estimate per-position drift for next step's active selection
            self._update_drift_proxy(K_full)
        else:
            # Sparse step
            active_recent = mask_state.unmasked_in_last_R(R=2) | mask_state.still_masked()
            active_magnitude = self._top_k_drift(self.k - len(active_recent))
            active_set = active_recent | active_magnitude
            
            # Gather scattered K/V for active set, pack into SRAM tile
            # Cached partition stays in HBM accessed via TMA blocks
            attn_out = self.mask_adaptive_attention_kernel(
                Q_indices=mask_state.still_masked(),
                KV_active_indices=active_set,
                KV_cache_handle=self.kv_cache,
            )
            # Commit newly computed K/V back to cache
            self.kv_cache.commit(active_set, K_new, V_new)
        
        # MLP packing — only for masked positions (predictions) and active updates
        mlp_out = self.mask_adaptive_mlp_kernel(
            input_indices=mask_state.still_masked() | active_set
        )
        
        # Predict at masked positions, unmask high-confidence ones
        new_predictions = self.model.head(attn_out, mlp_out)
        x_t_minus_1 = self._unmask_high_confidence(x_t, new_predictions)
        
        self.step_count += 1
        return x_t_minus_1
```

---

## 4. Theoretical analysis

### 4.1 Error sources and accumulation

```
   Per-step error from sparse approximation:
     err_step  =  ε_active-miss (positions wrongly excluded from active set)
                + ε_cache-stale (cached K/V is from before any drift since last refresh)
                + ε_softmax-combine (numerical error from two-pass online softmax)
   
   Cumulative error between anchors (M sparse steps):
     err_M  =  Σ err_step_i  ·  G_bidirectional^i
              where G is the per-step amplification from feedback through attention
   
   After anchor:
     err  =  0  (full dense recompute resets everything)
```

### 4.2 Why bidirectional cascade is the hard part

In AR speculative decoding (Leviathan et al. 2023), the rejection-sampling proof relies on conditional independence of token distributions given the prefix. In MDM, every position attends to every other position, so an error at position i propagates to outputs at positions j ≠ i, which become inputs at the next step.

Concretely: if K_t^[i] has error ε, then attention output at every position shifts by approximately ε · (attention weight to position i) · ‖V[i]‖. The net effect on K_{t+1}^[i] depends on how much the attention output at position i depends on its own incoming attention from elsewhere — a fixed-point structure.

**Tractable analysis path:** Bound the cascade growth factor G using attention sharpness (max attention probability) and value norm. Under a bound on attention sharpness (say, max prob < α < 1), the cascade is contractive and the M-step error is geometric in α^M.

The full proof is non-trivial but follows established techniques from sparse attention literature (e.g., Performers' attention approximation analysis, Reformer's locality-sensitive hashing analysis). Estimated complexity: 3–4 weeks of focused theoretical work.

### 4.3 What the bound buys us

A formal bound, even loose, gives:
- A principled way to set τ (magnitude threshold) given a target output error
- A principled way to set M (anchor period) given a target sequence-quality target
- Better error guarantees than DualDiffusion's heuristic verification (which has none)

Even if the bound is too loose to be quantitatively predictive, it's qualitatively informative — confirms that the method *can* preserve quality with appropriate hyperparameters, and identifies which assumptions are load-bearing.

---

## 5. Experiments

### 5.1 Models

| Model | Size | Architecture | Status |
|---|---|---|---|
| **LLaDA-8B-Instruct** (Nie et al. 2025) | 8B | Vanilla MHA, fully bidirectional, no GQA | Open-source, primary benchmark |
| **LLaDA-1.5** | 8B | Same | Open-source, secondary |
| **Dream-7B** (Ye et al. 2025) | 7B | Adapted from AR pretraining | Open-source, secondary |
| **MDLM** (Sahoo et al. 2024) | ~170M | Smaller MDM | Sanity check at small scale |
| **Mercury** (Inception Labs) | unknown | MDM, commercial | Best-effort if API access |

LLaDA-8B is the primary target because (a) it's the largest fully-open MDM, (b) it uses vanilla MHA with bigger K/V tensors than GQA models — more potential for kernel-level savings, (c) it's the standard baseline in FastDLLM and dKV-Cache.

### 5.2 Benchmarks

| Benchmark | Why | Expected speedup regime |
|---|---|---|
| **GSM8K** (5-shot, 8-shot, gen 256/512/1024) | Standard MDM benchmark; FastDLLM gets 27.6× here | Memory-bound; modest speedup over FastDLLM |
| **HumanEval** (0-shot, gen 256/512) | Code; FastDLLM gets only 3.7-4× here | **Compute-bound; biggest expected gain** |
| **MBPP** (3-shot) | Code; mid-difficulty | Moderate gain |
| **MATH** (4-shot) | Math reasoning | Moderate |
| **MMLU** (5-shot) | General reasoning, used by DualDiffusion | Memory-bound |
| **GPQA, MathVista** | dKV-Cache tested here too | Comparison baseline |
| **RULER 8K-32K** | Long-context; O(L²) cost matters most | **Largest expected gain** |

**Key strategic point:** the proposal's value-add is largest in **compute-bound** and **long-context** regimes where existing block caching saturates. HumanEval and RULER are the headline benchmarks. GSM8K and MMLU are sanity checks (we should at least match FastDLLM, hopefully 1.2-1.5× on top).

### 5.3 Baselines

```
   1. LLaDA-8B dense (no acceleration)              — quality ceiling, speed floor
   2. LLaDA-8B + FastDLLM PrefixCache               — best published method
   3. LLaDA-8B + FastDLLM DualCache + parallel      — best published method, full
   4. LLaDA-8B + dKV-Cache-Decode                   — closest to ours in spirit
   5. LLaDA-8B + dKV-Cache-Greedy                   — aggressive variant
   6. LLaDA-8B + DualDiffusion                      — algorithmic spec decoding
   7. Ours (kernel only, no spec decoding)          — isolate kernel contribution
   8. Ours + FastDLLM block cache                   — Layer 1 + Layer 2
   9. Ours + FastDLLM + DualDiffusion-style spec    — full stack (all 3 layers)
   10. Comparable AR LM (LLaMA-3-8B) with KV cache  — efficiency target
```

### 5.4 Hardware

- **H100-SXM5** (primary; matches Chipmunk and broader Hazy Research evaluation)
- **A100 80GB** (secondary; matches FastDLLM)
- **H200** (if access) (tertiary)
- **Skip Blackwell** unless surplus access; methodology should transfer

### 5.5 Metrics

```
   QUALITY
     - Task-specific accuracy (Pass@1, exact match, etc.)
     - Perplexity on held-out OWT subset
     - Token-level KL divergence vs full LLaDA
   
   SPEED
     - End-to-end latency (sec/generation)
     - Per-step kernel latency (attention + MLP)
     - Throughput (tokens/sec) at batch=1, batch=4, batch=16
     - Time-to-first-token (TTFT) and time-per-output-token (TPOT)
   
   MEMORY
     - Peak VRAM (the 2× memory of DualDiffusion is real and we should not compound it)
     - Cache size as fraction of model weights
   
   ROBUSTNESS
     - Speedup vs context length sweep (1K, 4K, 8K, 16K, 32K)
     - Quality vs anchor period M (sweep M ∈ [3, 20])
     - Quality vs active-set fraction (sweep 0.1 to 0.5)
     - Long-sequence stability (no degradation at 32K)
```

### 5.6 Ablations

```
   1. Active selection signal:    recency only | magnitude only | hybrid
   2. Anchor period M:            3, 5, 7, 10, 15, 20
   3. Active set fraction:        10%, 20%, 30%, 50% of L
   4. Token reordering:           none | mask-state | learned
   5. Kernel components:          dense | gather-only | gather+pack | full
   6. RoPE handling:              pre-rotate vs post-gather rotate
   7. Spec decoding composition:  off | DualDiffusion-style
   8. Hardware:                   A100 vs H100 vs H100+TMA
   9. Sequence length:            512, 1K, 2K, 4K, 8K, 16K, 32K
   10. MLP-only vs Attention-only: which gives more speedup independently
```

### 5.7 Hypotheses to test

H1. **Magnitude beats recency for active selection** at fixed compute budget. (Test: ablation 1 above on GSM8K and RULER.)

H2. **Compute-bound regime sees largest gain.** Speedup ratio (ours vs FastDLLM) is highest on HumanEval and RULER, lowest on GSM8K-1024. (Test: speedup-by-benchmark plot.)

H3. **Hopper-specific features matter.** Speedup on H100 vs A100 isn't just a constant factor; the gather/pack overhead is meaningfully lower on Hopper due to `cp.async` + TMA. (Test: A100 vs H100 results table.)

H4. **The error bound is predictive.** Theoretical setting of M and τ achieves the predicted accuracy/speed tradeoff within 10–15%. (Test: theoretical-vs-empirical Pareto curve.)

H5. **Stacking is genuinely composable.** Layer 2 (this work) gives ~1.3-1.5× on top of Layer 1 (FastDLLM); Layer 3 (spec decoding) gives ~1.3-1.5× on top of Layer 2. (Test: factor-decomposition table.)

---

## 6. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **FastDLLM's 27.6× saturates the speedup ceiling.** Adding kernel-level work on top yields only 1.05-1.15× — not worth a paper. | **High** | Focus headline metrics on compute-bound regimes where FastDLLM gets only 3-4×. RULER and HumanEval are honest demonstrations. If the speedup over FastDLLM is genuinely <1.2× across the board, pivot to characterization paper ("first MDM kernel work on Hopper") rather than speedup paper. |
| **Bidirectional cascade is worse in text than images.** Errors propagate to wrong tokens and the bound is not tight. | High | (a) Tighter anchor period M=3-5; (b) keep all currently-masked positions always-active (never approximate predictions themselves); (c) accept that the bound may be loose and lean on empirical robustness. |
| **Position embeddings break the permutation trick.** RoPE bakes position into Q, K, can't reorder freely. | High | Apply RoPE *after* gather (carry position indices alongside packed rows). dKV-Cache's appendix shows this is feasible with constant overhead. |
| **Active-set selection is itself expensive.** Top-k over L positions every step eats the savings. | Medium | Use cheap proxy: norm of single-layer feature, or early-exit head from layer 4 of a 32-layer model. Amortize selection across multiple steps. Chipmunk's "approximate top-k" gives 3.42× speedup over PyTorch top-k. |
| **Combined softmax (active + cached partitions) has numerical issues.** | Medium | FlashAttention's two-pass online softmax is the established solution. Port carefully. |
| **Cache HBM layout is complex.** Need fast block reads (cached) and scattered writes (active commit). | Medium | Start v1 with no caching at all (just packed compute); add caching as v2. Reduces engineering risk. |
| **Mercury is closed-source; can't compare.** | Low | LLaDA is the public-MDM standard. Mercury is "would be nice." |
| **Someone in Dan's group is already doing this.** | Unknown | First step before deep investment: ask Dan directly. Even if so, the magnitude-selection or theoretical-bound angle could be a clean orthogonal contribution. |
| **Eso-LMs becomes the dominant approach and training-free MDM acceleration becomes irrelevant.** | Low-medium | Different niche: Eso-LMs needs full retraining and only exists at small scale. As long as LLaDA, Mercury, Dream are the deployed models, training-free acceleration matters. |

---

## 7. Timeline (6-month milestone plan)

```
Month 1 — Reproduce baselines.
  Set up LLaDA-8B inference. Reproduce FastDLLM, dKV-Cache, DualDiffusion
  on MMLU, GSM8K, HumanEval. Confirm published numbers within 5%.
  Establish quality ceiling and current SOTA at multiple speedup tiers.
  Run all baselines on H100 (which nobody has done) — that itself is a
  useful intermediate result.

Month 2 — Kernel v1: gather + pack + dense GEMM, no caching.
  Write the gather-pack-wgmma kernel in ThunderKittens. No caching yet —
  recompute K/V for all positions every step, but use the packed-compute
  pattern. Verify bit-correctness against dense LLaDA. Should match
  numerically while being modestly faster (just from the packed compute).
  Expected speedup: 1.3-1.8× per step.

Month 3 — Kernel v2: K/V cache + delta updates.
  Add the HBM cache layout, scattered commit, two-partition softmax
  combine. Implement re-anchor scheduling.
  Expected speedup at this point: 2.5-3× per step at no quality cost on
  MMLU. Hyperparameter sweeps on M and active fraction.

Month 4 — Active selection ablation + composition.
  Implement and compare recency-only, magnitude-only, hybrid signals.
  Compose with FastDLLM (Layer 1+2 stack) and verify composability.
  All ablations from §5.6.

Month 5 — Theoretical analysis + spec decoding composition.
  Write the formal error bound. Plug into DualDiffusion-style spec
  decoding (Layer 3). Verify quality on GSM8K (where DualDiffusion failed).
  Run the long-context benchmarks (RULER 8K-32K).

Month 6 — Paper writing + final ablations + open-source release.
  Polish ThunderKittens code for release. Final ablations including
  hardware comparison (A100 vs H100). Primary submission: ICLR 2027
  workshops (SLLM "Sparsity in LLMs" and/or DeLTa "Deep Generative
  Models") — deadline ~Feb 2027, ~50–65% acceptance, non-archival, good
  fit for kernel + sparse-attention story on diffusion LMs. Stretch
  submission: MLSys 2027 main track if numbers and writing are mature
  enough by the March 2027 deadline.
```

The critical path is Months 2–3 (kernel v1 → v2). If the kernel work falls behind, Months 4–5 compress; theoretical analysis and spec-decoding composition can both run in parallel with engineering.

---

## 8. Why this proposal fits Dan Fu's group specifically

**Direct alignment with Dan's published work:**
- Dan is **senior author on Chipmunk** (Silveria, Govande, Fu 2025). The methodology being ported is literally his group's.
- ThunderKittens (Spector, Fu et al. 2024) is the kernel DSL we'd use.
- FlashAttention (Dao, Fu et al. 2022) and FlashFFTConv (Fu et al. 2024) establish the IO-aware kernel design pattern this proposal extends.
- M2/Monarch Mixer (Fu et al. 2023) is structured-sparsity counterpart to this proposal's dynamic sparsity — shared philosophical lineage.

**Adjacent expertise (training-side) that the group has but hasn't applied to MDMs:**
- Linear/structured attention (Hyena, M2)
- Long-context (FlashFFTConv)
- Speculative decoding adjacent (work on efficient inference more broadly)

**Cross-pollination that fits the group's track record:**
- Hazy Research lineage of "hardware-aware acceleration with theoretical grounding" — same template applied to a fresh model class.
- Compositional with concurrent work (FastDLLM, dKV-Cache, DualDiffusion) rather than competing with them. Strong narrative.
- MLSys / ICLR is the natural venue and matches Dan's publication record.

**Practical advantages of being in this group:**
- Direct access to Chipmunk authors (Silveria at Together, Govande at Stanford) for technical advisory.
- ThunderKittens infrastructure already in place.
- H100 access via Together AI partnership.
- Existing paper-writing pipeline and reviewer relationships.

---

## 9. Open questions for advisor conversation

1. **Is anyone in your group, Sandy Research, or Inception Labs already doing this?** Sanity check before deep investment. Given Chipmunk is recent (June 2025) and MDM literature even more so, the field is small enough that direct asking saves months.

2. **Mercury access?** Inception Labs sometimes provides academic API access. Would broaden empirical evaluation significantly. Worth asking.

3. **Co-authorship with Chipmunk authors?** Silveria/Govande have the kernel craft; involving them turns this into a stronger paper with name recognition. Joint venture between Hazy Research / Sandy Research / MDM-LM communities.

4. **Venue framing — workshop-first vs main-track?** My current plan is to target an **ICLR 2027 workshop** as the primary submission (SLLM "Sparsity in LLMs" or DeLTa "Deep Generative Models"; ~Feb 2027 deadline, ~50–65% acceptance, non-archival). This lets the work get external feedback while results are still maturing, without burning the main-track novelty. MLSys 2027 main track (March 2027 deadline) is a stretch goal if the kernel numbers and theoretical bound land cleanly. Heavy kernels lean MLSys; theoretical bound + algorithmic spec composition leans ICLR — open question whether the story is strong enough by March for the main track, or whether ES-FoMo @ ICML 2027 (May deadline) is a better second-attempt venue.

5. **Distilled drafter as separate contribution?** The drafter design space (Medusa, EAGLE, self-spec, lookahead) is large enough to be its own paper. Could split into two papers if scope balloons — first this kernel + algorithmic paper, then a follow-up on drafter design specifically.

6. **Is the theoretical analysis worth pursuing seriously?** Even a loose bound is novel for MDMs. But it's 3-4 weeks of focused work. If you'd rather see kernel speedup first and add theory later, that's a reasonable order-of-operations call.

---

## 10. References (verified)

```
[Goyal2026]      Goyal, Patel, Mittal, Laxman.
                 DualDiffusion: A Speculative Decoding Strategy for Masked
                 Diffusion Models. arXiv:2604.05250, 2026.

[Nie2025]        Nie, Zhu, You, Zhang, Ou, Hu, Zhou, Lin, Wen, Li.
                 Large Language Diffusion Models. (LLaDA)
                 arXiv:2502.09992, NeurIPS 2025.

[Sahoo2024]      Sahoo, Arriola, Schiff, Gokaslan, Marroquin, Chiu, Rush, Kuleshov.
                 Simple and Effective Masked Diffusion Language Models. (MDLM)
                 NeurIPS 2024.

[Wu2025]         Wu, Zhang, Xue, Liu, Diao, Zhu, Luo, Han, Xie.
                 Fast-dLLM: Training-free Acceleration of Diffusion LLM by
                 Enabling KV Cache and Parallel Decoding.
                 arXiv:2505.22618, 2025.

[Ma2025]         Ma, Yu, Fang, Wang.
                 dKV-Cache: The Cache for Diffusion Language Models.
                 arXiv:2505.15781, 2025.

[Yang2025]       Sahoo, Yang, Akhauri, Liu, Singh, Cheng, Liu, Xing, Thickstun, Vahdat.
                 Esoteric Language Models. (Eso-LMs)
                 arXiv:2506.01928, 2025.

[Arriola2025]    Arriola, Gokaslan, Chiu, Yang, Qi, Han, Sahoo, Kuleshov.
                 Block Diffusion: Interpolating Between Autoregressive and
                 Diffusion Language Models. (BD3-LMs)
                 arXiv:2503.09573, 2025.

[Silveria2025]   Silveria, Govande, Fu.
                 Chipmunk: Training-Free Acceleration of Diffusion Transformers
                 with Dynamic Column-Sparse Deltas.
                 arXiv:2506.03275, 2025.

[Spector2024]    Spector, Singhal, Arora, Fu, Re.
                 ThunderKittens: Simple, Fast, and Adorable AI Kernels.
                 arXiv:2410.20399, 2024.

[Dao2022]        Dao, Fu, Ermon, Rudra, Re.
                 FlashAttention: Fast and Memory-Efficient Exact Attention
                 with IO-Awareness. NeurIPS 2022.

[Fu2024]         Fu, Kumbong, Nguyen, Re.
                 FlashFFTConv. ICLR 2024.

[Fu2023]         Fu, Arora, Grogan, Johnson, Eyuboglu, Thomas, Spector,
                 Poli, Rudra, Re.
                 Monarch Mixer (M2). NeurIPS 2023.

[Leviathan2023]  Leviathan, Kalman, Matias.
                 Fast Inference from Transformers via Speculative Decoding.
                 arXiv:2211.17192, 2023.

[Chen2023]       Chen, Borgeaud, Mensch, Sutskever, Sifre, Vinyals et al.
                 Accelerating Large Language Model Decoding with Speculative
                 Sampling. arXiv:2302.01318, 2023.

[Austin2021]     Austin, Johnson, Ho, Tarlow, Van Den Berg.
                 Structured Denoising Diffusion Models in Discrete State-Spaces.
                 (D3PM) NeurIPS 2021.

[Ye2025]         Ye, Xie, Zheng, Gao, Wu, Jiang, Li, Kong.
                 Dream 7B. 2025.
```

---

## 11. Honest summary

The opening is real, smaller than I first claimed, but defensible.

- **Smaller than claimed**: FastDLLM gets 27.6× algorithmically. The "8-12× over LLaDA" target was wrong.
- **Defensible**: The kernel-level work specifically — `cp.async` gather, SRAM packing with `wgmma`, magnitude-based active selection, formal error bound — has not been done for MDMs and is explicitly invited by dKV-Cache's authors as future work. Dan Fu literally wrote the Chipmunk paper.
- **Best-positioned target**: 1.3-2× on top of FastDLLM in compute-bound and long-context regimes (HumanEval, RULER). Stacked total speedup over plain LLaDA: 30-50× on long-generation, 8-15× on compute-bound — competitive with AR LMs of the same scale.

The methodology is well-understood, the engineering work is substantial but bounded, and the alignment with Dan's group is structurally perfect (he's a Chipmunk co-author, the kernel DSL is his group's, the methodology pattern matches three of his prior papers).

Risk: FastDLLM saturates the speedup ceiling more than expected and there's nothing left to claim beyond a characterization paper. Mitigation: characterize first (Month 1), pivot framing if needed.

If executed: a clean MLSys / ICLR paper with stacked speedup, theoretical bound, open-source kernel release. Strong CV item. Real research contribution.

Next concrete step: a 1-page summary of this for Dan, asking the open questions in §9, plus a quick check whether anyone's already on this.
