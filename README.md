<div align="center">
  <a href="tech_report.pdf"><img width="80%" src="figures/banner.png"></a>
</div>

<!-- # Muon is Scalable For LLM Training -->

<div align="center">
  <a href="tech_report.pdf"><img src="figures/logo.png" height="16" width="16" style="vertical-align:middle"><b> Tech Report</b></a>  |  
  <a href="https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="16" width="16" style="vertical-align:middle"><b> HuggingFace</b></a>
</div>

## Abstract

We introduce Kimi Linear, a hybrid linear attention architecture that, for the first time, outperforms full attention under fair comparisons across various scenarios—including short-context, long-context, and reinforcement learning (RL) scaling regimes. At its core lies Kimi Delta Attention (KDA), an expressive linear attention module that extends [Gated DeltaNet](https://arxiv.org/abs/2412.06464) with a finer-grained gating mechanism, enabling more effective use of limited finite-state RNN memory. 
Our bespoke chunkwise algorithm achieves high hardware efficiency through a specialized variant of the *Diagonal-Plus-Low-Rank* (DPLR) transition matrices, which substantially reduces computation compared to the general DPLR formulation while remaining more consistent with the classical delta rule.

We pretrain a Kimi Linear model with 3B activated parameters and 48B total parameters, based on a layerwise hybrid of KDA and Multi-Head Latent Attention (MLA). Our experiments show that with an identical training recipe, Kimi Linear outperforms full MLA with a sizeable margin across all evaluated tasks, while reducing KV cache usage by up to 75\% and achieving up to $6\times$ decoding throughput for a 1M context. These results demonstrate that Kimi Linear can be a drop-in replacement for full attention architectures with superior performance and efficiency, including tasks with longer input and output lengths.

We open-source the [KDA kernel](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda), and release the [pre-trained](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base) and [instruction-tuned](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) model checkpoints.

## Contributions

- **Kimi Delta Attention (KDA):** a linear attention mechanism that refines the gated delta rule with improved recurrent memory management and hardware efficiency.
- **The Kimi Linear architecture:** a hybrid design adopting a 3:1 KDA-to-global attention ratio, reducing memory footprint while surpassing full-attention quality.
- **Fair empirical validation at scale:** through 1.4T token training runs, Kimi Linear outperforms full attention and other baselines in short/long context and RL-style evaluations, with full release of kernels, vLLM integration, and checkpoints.
