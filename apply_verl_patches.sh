#!/bin/bash
set -e

echo "Applying verl compatibility patches..."

# Apply patches to verl submodule source
cd verl

echo "  - Fixing fsdp_workers.py for flash-attn compatibility..."
if patch -p1 --dry-run --silent < ../patches/verl-fsdp-workers.patch 2>/dev/null; then
    patch -p1 < ../patches/verl-fsdp-workers.patch
    echo "    ✓ Applied fsdp_workers.py patch"
else
    echo "    ⚠ Patch already applied (skipping)"
fi

echo "  - Fixing dp_actor.py flash-attn imports..."
if patch -p1 --dry-run --silent < ../patches/verl-dp-actor.patch 2>/dev/null; then
    patch -p1 < ../patches/verl-dp-actor.patch
    echo "    ✓ Applied dp_actor.py patch"
else
    echo "    ⚠ Patch already applied (skipping)"
fi

echo "  - Fixing dp_critic.py flash-attn imports..."
if patch -p1 --dry-run --silent < ../patches/verl-dp-critic.patch 2>/dev/null; then
    patch -p1 < ../patches/verl-dp-critic.patch
    echo "    ✓ Applied dp_critic.py patch"
else
    echo "    ⚠ Patch already applied (skipping)"
fi

echo "  - Fixing vLLM version detection..."
if patch -p1 --dry-run --silent < ../patches/verl-vllm-init.patch 2>/dev/null; then
    patch -p1 < ../patches/verl-vllm-init.patch
    echo "    ✓ Applied vllm __init__.py patch"
else
    echo "    ⚠ Patch already applied (skipping)"
fi

echo "  - Fixing vLLM 0.10.1 compatibility..."
if patch -p1 --dry-run --silent < ../patches/verl-vllm-rollout.patch 2>/dev/null; then
    patch -p1 < ../patches/verl-vllm-rollout.patch
    echo "    ✓ Applied vllm_rollout.py patch"
else
    echo "    ⚠ Patch already applied (skipping)"
fi

cd ..

echo ""
echo "Reinstalling verl in editable mode to apply patches to installed package..."
uv pip install -e verl --no-build-isolation

echo ""
echo "✓ All patches applied successfully and verl reinstalled"
echo ""
echo "Note: These patches fix:"
echo "  - Flash attention compatibility (removes flash_attn imports)"
echo "  - vLLM 0.10.1 API compatibility"
echo "  - vLLM version detection for proper mode selection"
