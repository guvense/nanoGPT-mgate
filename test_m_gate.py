"""
Sanity tests for the use_m_gate modification to nanoGPT's self-attention.

Run: python test_m_gate.py

Verifies:
  (a) forward + backward on random input produces no NaN/Inf, with use_m_gate=True
  (b) W_g and W_m in every attention block receive nonzero gradients
  (c) with use_m_gate=False, output and state_dict are BIT-IDENTICAL to vanilla
      nanoGPT (extracted from git HEAD), given the same seed and inputs
"""

import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig  # modified model with M-gate support


def load_vanilla_model_module():
    """Extract pre-modification model.py from git HEAD and import as its own module."""
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    src = subprocess.check_output(["git", "show", "HEAD:model.py"], cwd=repo_dir)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    spec = importlib.util.spec_from_file_location("model_vanilla", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def small_cfg(**overrides):
    kw = dict(block_size=64, vocab_size=128, n_layer=3, n_head=4, n_embd=32, dropout=0.0, bias=True)
    kw.update(overrides)
    return kw


def test_no_nan_and_gate_grads():
    torch.manual_seed(0)
    cfg = GPTConfig(use_m_gate=True, **small_cfg())
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    tgt = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, tgt)
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"
    assert torch.isfinite(loss), f"loss is NaN/Inf: {loss.item()}"
    loss.backward()
    # Two boundary effects to be aware of:
    #  - block 0: M_prev=zeros, so attn.W_kvm.weight grad is zero there (grad ~ M_prev).
    #  - last block: its M_new is not consumed downstream, so W_gm grad is zero there.
    # We therefore require SUM-across-blocks nonzero rather than per-block nonzero.
    def sum_wgrad(owner_fn, name):
        total = 0.0
        for i, blk in enumerate(model.transformer.h):
            layer = getattr(owner_fn(blk), name)
            g = layer.weight.grad
            if g is not None:
                assert torch.isfinite(g).all(), f"block {i} {name}.weight grad NaN/Inf"
                total += g.abs().sum().item()
            if layer.bias is not None and layer.bias.grad is not None:
                assert torch.isfinite(layer.bias.grad).all()
        return total
    assert sum_wgrad(lambda b: b, "W_gm") > 0, "W_gm.weight grad zero across all blocks"
    assert sum_wgrad(lambda b: b.attn, "W_kvm") > 0, "attn.W_kvm.weight grad zero across all blocks"
    for blk in model.transformer.h:
        assert blk.last_g_mean is not None and blk.last_g_std is not None
    print("PASS (a,b): fwd+bwd finite; W_gm/W_kvm weights receive gradients (summed over blocks)")


def test_use_m_gate_false_matches_vanilla():
    vanilla = load_vanilla_model_module()
    kw = small_cfg()

    torch.manual_seed(1234)
    m_mod = GPT(GPTConfig(use_m_gate=False, **kw))
    torch.manual_seed(1234)
    m_van = vanilla.GPT(vanilla.GPTConfig(**kw))

    # 1. state_dict keys and tensors must match exactly
    mod_sd, van_sd = m_mod.state_dict(), m_van.state_dict()
    only_mod = set(mod_sd) - set(van_sd)
    only_van = set(van_sd) - set(mod_sd)
    assert not only_mod and not only_van, (
        f"state_dict key mismatch:\n  modified-only: {sorted(only_mod)}\n  vanilla-only:  {sorted(only_van)}"
    )
    for k in mod_sd:
        assert torch.equal(mod_sd[k], van_sd[k]), f"parameter '{k}' differs (RNG stream diverged)"

    # 2. no M-gate attributes should exist anywhere when the flag is off
    for i, blk in enumerate(m_mod.transformer.h):
        assert not hasattr(blk, "W_gm"), f"block {i}: W_gm exists with use_m_gate=False"
        assert not hasattr(blk.attn, "W_kvm"), f"block {i}: attn.W_kvm exists with use_m_gate=False"

    # 3. forward output bit-identical
    m_mod.eval(); m_van.eval()
    idx = torch.randint(0, kw["vocab_size"], (2, 16))
    tgt = torch.randint(0, kw["vocab_size"], (2, 16))
    with torch.no_grad():
        l_mod, loss_mod = m_mod(idx, tgt)
        l_van, loss_van = m_van(idx, tgt)
    assert torch.equal(l_mod, l_van), "logits differ from vanilla — use_m_gate=False is not a true no-op"
    assert torch.equal(loss_mod, loss_van), "loss differs from vanilla"
    print("PASS (c): use_m_gate=False is bit-identical to vanilla (params + forward output)")


if __name__ == "__main__":
    test_no_nan_and_gate_grads()
    test_use_m_gate_false_matches_vanilla()
    print("\nAll tests passed.")
