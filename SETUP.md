# Setup Instructions

## Quick Start

After cloning this repository, run:

```bash
# 1. Sync dependencies (installs verl and all dependencies)
uv sync

# 2. Apply verl compatibility patches
./apply_verl_patches.sh
```

That's it! The patches fix compatibility issues with vLLM 0.10.1 and flash-attn.

## What the Patches Fix

The `apply_verl_patches.sh` script automatically applies patches to the verl library that fix:

1. **Flash Attention Compatibility**
   - Removes hardcoded `flash_attention_2` usage
   - Uses `eager` attention instead (configurable via config files)
   - Fixes actor, critic, and reward model initialization

2. **vLLM 0.10.1 Compatibility**
   - Fixes version detection (uses proper semantic versioning)
   - Removes deprecated `model_hf_config` parameter
   - Enables SPMD mode for vLLM > 0.6.3

3. **Configuration Updates**
   - Context length adjusted for Qwen2.5-Coder-3B (32K tokens)
   - Eager execution enabled for stability
   - Ray environment variables configured

## Manual Setup (if script fails)

If the automatic patching fails, you can manually apply patches:

```bash
cd verl
patch -p1 < ../patches/verl-fsdp-workers.patch
patch -p1 < ../patches/verl-vllm-init.patch
patch -p1 < ../patches/verl-vllm-rollout.patch
cd ..
```

## Weights & Biases (Optional)

If you want to use wandb for logging:

```bash
wandb login
```

Otherwise, wandb is configured in the trainer configs and will work with console logging only.
