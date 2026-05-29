# WaterSIC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-2603.04956-b31b1b.svg)](https://arxiv.org/abs/2603.04956)

Information-theoretically (near) optimal post-training quantization for LLM linear layers.
WaterSIC allocates different quantization rates to different in-channels of the weight matrix via
waterfilling, achieving a rate gap of at most 0.255 bits to the IT limit. Establishes new SOTA on
Llama and Qwen3 from 1 to 4 bits.

Paper: [*WaterSIC: information-theoretically (near) optimal linear layer quantization*](https://arxiv.org/abs/2603.04956) (ICML 2026)

## Install

```bash
git clone https://github.com/egorlifar/w-quant-new.git
cd w-quant-new
pip install -e .            # adds the `watersic-*` console scripts
# optional eval extras: pip install -e ".[eval]"
```

Requires Python ≥ 3.10, CUDA 12.x PyTorch (auto-selected by `uv` if you use it).

Set `QUANT_BUCKET` to point at model checkpoints and run output storage:

```bash
export QUANT_BUCKET=/path/to/quant-bucket
```

### Hardware

| Model | Recommended GPUs |
|---|---|
| Llama-3.2-1B / Llama-2-7B / Llama-3-8B / Qwen3-8B | 1× A100 80GB or H100 80GB |
| Llama-3-70B | 4× H100 80GB |

Llama-3-70B requires both 4-way tensor parallelism (see [70B section](#llama-3-70b-4-gpus-tensor-parallel)) and the H100 memory headroom for the `w2`-input Hessian (28672² float64 ≈ 6 GB per rank). Smaller GPUs are unlikely to fit.

## Model setup

Place Llama checkpoints (Meta `.pth` format) under `$QUANT_BUCKET/`:

```
$QUANT_BUCKET/
  Llama-3-8B/            # consolidated.00.pth + params.json + tokenizer.model
  Llama3.2-1B/           # same format
  Llama-2-7B/            # same format
  Llama-3-70B-4shard/    # 4 shards (use scripts/shard_checkpoint.py to create)
```

For Qwen3-8B, the model is loaded from HuggingFace automatically (or cached locally at `$QUANT_BUCKET/Qwen3-8B/` if downloaded via `huggingface-cli download Qwen/Qwen3-8B`).

## Quick start

### Quantize end-to-end (single command)

```bash
watersic-quantize-and-finetune \
  --model 3-8B --method zsic --target_rate 3.0 \
  --zsic_binary_search --rate_control \
  --qronos --residual_compensation \
  --qronos_adapt --attn_weighted_qkv --attn_weighted_adapt_eps_joint \
  --finetune
```

This runs `watersic-quantize` followed by `watersic-finetune` on the resulting run directory.

### Or: quantize / finetune / eval in separate steps

**Quantize** (Llama-3-8B at 3 bits):
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  -m scripts.run_pipeline_job \
  --model 3-8B --method zsic --target_rate 3.0 \
  --zsic_binary_search --rate_control \
  --qronos --residual_compensation \
  --qronos_adapt --attn_weighted_qkv --attn_weighted_adapt_eps_joint
```

**Evaluate** (PPL + KL):
```bash
python -m scripts.run_eval_job --run_dir $QUANT_BUCKET/quant_runs/3-8B/<run_id>
```

**Finetune** (WaterSIC-FT via KL distillation):
```bash
python scripts/finetune_zsic.py \
  --model_name 3-8B --run_dir $QUANT_BUCKET/quant_runs/3-8B/<run_id> --gpu 0
```

## Reproducing paper results

### Llama-3.2-1B (rate sweep 1–4 bits)
```bash
python -m scripts.run_quant_sweep \
  --model 3.2-1B --method zsic \
  --rates "1.0,1.5,2.0,2.5,3.0,3.18,3.5,3.76,4.0" \
  --qronos --residual_compensation \
  --qronos_adapt --attn_weighted_qkv --attn_weighted_adapt_eps_joint
```

### Qwen3-8B
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  -m scripts.run_pipeline_job \
  --model qwen3-8B --method zsic --target_rate 2.125 \
  --zsic_binary_search --rate_control \
  --qronos --residual_compensation \
  --qronos_adapt --attn_weighted_qkv --attn_weighted_adapt_eps_joint \
  --zero_out_rows "6.w1:5723,8518;6.w3:5723,8518;16.w1:2271,1875;16.w3:2271,1875"
```

### Llama-3-70B (4 GPUs, tensor parallel)

70B quantization runs over 4 tensor-parallel ranks, so the base checkpoint must first be resharded into 4 pieces. We ran on **4× H100 80GB**; smaller GPUs are unlikely to fit the per-rank Hessian + activation cache for `w2`.

First, shard the checkpoint and precompute Hessians (one-time):
```bash
# Shard 70B checkpoint from 8 shards (Meta release) into 4 — required for 4-GPU TP
python scripts/shard_checkpoint.py \
  --input_dir $QUANT_BUCKET/Llama-3-70B --output_dir $QUANT_BUCKET/Llama-3-70B-4shard --num_shards 4

# Precompute Hessians + attention importance (saves ~50% per-run time)
torchrun --nproc_per_node=4 -m quant_layerwise.precompute \
  --model_name 3-70B-4s \
  --output_dir $QUANT_BUCKET/precomputed/3-70B-4s/wikitext2_s2048_seed42
```

Then quantize:
```bash
OMP_NUM_THREADS=48 torchrun --nproc_per_node=4 \
  -m scripts.run_pipeline_job \
  --model 3-70B-4s --method zsic --target_rate 4.0 --layer_end 80 \
  --zsic_binary_search --rate_control \
  --qronos --residual_compensation \
  --qronos_adapt --attn_weighted_qkv --attn_weighted_adapt_eps_joint \
  --hessian_batch_size 32 --replay_batch_size 12 --w2_batch_size 4 \
  --precomputed_dir $QUANT_BUCKET/precomputed/3-70B-4s/wikitext2_s2048_seed42
  # On large models the joint eps search dominates runtime. To speed it up:
  #   --coord_adapt_q_eps_steps 6   fewer golden-section steps (default 10)
  #   --coord_adapt_a_eps_steps 0   search only q_eps, hold a_eps fixed (default 10)
```

### Multi-GPU finetuning (70B)

Same 4× H100 setup as the quantize step. For 70B we use a lower peak LR than the small-model default (`5e-5` vs `5e-4`) — the larger student is more sensitive to the rescaler perturbations during early epochs and converges more cleanly with the gentler schedule:

```bash
torchrun --nproc_per_node=4 scripts/finetune_zsic.py \
  --model_name 3-70B-4s --run_dir $QUANT_BUCKET/quant_runs/3-70B-4s/<run_id> \
  --batch_size 2 --epochs 4 --lr 5e-5 --min_lr 5e-6
```

### GPTQ baseline
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  -m scripts.run_pipeline_job \
  --model 3.2-1B --method gptq --target_rate 4.0 \
  --hessian_batch_size 16 --percdamp 0.1
```

## Project structure

```
w-quant-new/
  quant_layerwise/
    pipeline.py            # main quantization loop
    methods/zsic.py        # WaterSIC core (ZSIC + waterfilling + rescalers)
    methods/gptq.py        # GPTQ baseline
    finetune.py            # KL distillation finetuning (WaterSIC-FT)
    precompute.py          # precompute Hessians for multi-GPU
    hessian_runtime.py     # activation covariance collection
    qronos_stats.py        # drift correction statistics
    data.py                # WikiText-2 / C4 / RedPajama loading
    eval.py                # PPL and KL evaluation
    partial_model.py       # apply quantized weights to a model in-place
    storage/artifacts.py   # LayerArtifact + RunManifest
  scripts/
    run_pipeline_job.py        # quantize one model (paper-repro entry point)
    run_eval_job.py            # evaluate one run
    finetune_zsic.py           # finetune one run (single or multi-GPU)
    run_quant_sweep.py         # rate sweep across GPUs
    run_eval_sweep.py          # eval a sweep
    quantize_and_finetune.py   # one-shot quantize → finetune wrapper
    shard_checkpoint.py        # reshape Meta-format checkpoints
    merge_checkpoint.py        # inverse of shard_checkpoint
    plot_activation_mse.py     # optional --plot_activation_mse hook called by pipeline.py
  examples/paper/              # diagnostic plots and exploratory analyses (see examples/README.md)
  parallel/                    # FairScale tensor-parallel Llama loader + Qwen3 adapter
  lm_bench/                    # lm-eval-harness wrapper for zero-shot tasks
  tests/                       # smoke tests (CI)
```

## Supported models

| Name | Layers | Shards | Registry key |
|---|---|---|---|
| Llama-3.2-1B | 16 | 1 | `3.2-1B` |
| Llama-2-7B | 32 | 1 | `2-7B` |
| Llama-3-8B | 32 | 1 | `3-8B` |
| Qwen3-8B | 36 | 1 | `qwen3-8B` |
| Llama-3-70B | 80 | 4 | `3-70B-4s` |

## Citation

```bibtex
@inproceedings{lifar2026watersic,
  title  = {WaterSIC: information-theoretically (near) optimal linear layer quantization},
  author = {Lifar, Egor and Savkin, Semyon and Ordentlich, Or and Polyanskiy, Yury},
  booktitle = {International Conference on Machine Learning (ICML)},
  year   = {2026},
}
```
