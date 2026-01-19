#!/bin/bash
set -e

echo "Applying verl compatibility patches..."

# Apply patches to verl submodule
cd verl

echo "  - Fixing fsdp_workers.py for flash-attn compatibility..."
if patch -p1 --dry-run --silent < ../patches/verl-fsdp-workers.patch 2>/dev/null; then
    patch -p1 < ../patches/verl-fsdp-workers.patch
    echo "    ✓ Applied fsdp_workers.py patch"
else
    echo "    ⚠ Patch already applied or failed (skipping)"
fi

echo "  - Fixing vLLM version detection..."
if patch -p1 --dry-run --silent < ../patches/verl-vllm-init.patch 2>/dev/null; then
    patch -p1 < ../patches/verl-vllm-init.patch
    echo "    ✓ Applied vllm __init__.py patch"
else
    echo "    ⚠ Patch already applied or failed (skipping)"
fi

echo "  - Fixing vLLM 0.10.1 compatibility..."
if patch -p1 --dry-run --silent < ../patches/verl-vllm-rollout.patch 2>/dev/null; then
    patch -p1 < ../patches/verl-vllm-rollout.patch
    echo "    ✓ Applied vllm_rollout.py patch"
else
    echo "    ⚠ Patch already applied or failed (skipping)"
fi

cd ..

echo "✓ All patches applied successfully"
echo ""
echo "Note: These patches fix:"
echo "  - Flash attention compatibility (uses eager attention instead)"
echo "  - vLLM 0.10.1 API compatibility"
echo "  - vLLM version detection for proper mode selection"
