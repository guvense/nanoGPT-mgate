"""
Phase-A scale timing benchmark: does K/V-bias wall-clock overhead shrink at
larger n_embd?

At small n_embd (112 in our main experiments), K/V added ~18% per-iter cost on
CUDA-T4. Theory says the ratio should stay roughly constant if compute is
FLOP-bound, or shrink if kernel-launch overhead dominates at small sizes.

This script builds fresh models at various n_embd sizes (no training, no
checkpoints needed), runs forward+backward+optimizer.step() a few times, and
prints a table of overhead-vs-size. ~5 minutes on a T4, $0.

Usage:
    python analyze_scale.py                              # default sizes
    python analyze_scale.py --sizes 128 256 512 768 1024 # custom
"""

import argparse
import time

import torch

from model import GPT, GPTConfig


def build_model(n_embd, n_head, n_layer, use_m_gate, device, dtype, vocab_size=50304, block_size=256):
    cfg = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=0.0, bias=False,
        use_m_gate=use_m_gate,
    )
    model = GPT(cfg).to(device=device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    return model


def time_forward_backward(model, x, y, n_warmup, n_iter, device, ctx):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(n_warmup):
        opt.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        opt.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        opt.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.time() - t0) / n_iter * 1000  # ms/iter


def pick_n_head(n_embd, target_head_dim=64):
    """Choose n_head so head_dim is close to target and n_embd divides evenly."""
    n_head = max(1, n_embd // target_head_dim)
    while n_head > 0 and n_embd % n_head != 0:
        n_head -= 1
    return max(1, n_head)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[128, 192, 256, 384, 512, 768])
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--n_iter", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--vocab_size", type=int, default=50304)
    args = p.parse_args()

    device = args.device
    if not device.startswith("cuda") and not device.startswith("mps"):
        args.dtype = "float32"
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    if device.startswith("cuda"):
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    print(f"Device: {device} | dtype: {args.dtype}")
    print(f"Config: n_layer={args.n_layer}, batch_size={args.batch_size}, "
          f"block_size={args.block_size}")
    print(f"Timing: {args.n_warmup} warmup + {args.n_iter} timed iters per model\n")

    header = f"{'n_embd':>7} {'n_head':>7} {'head_dim':>9} {'params':>10} {'vanilla':>10} {'K/V':>10} {'overhead':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for n_embd in args.sizes:
        n_head = pick_n_head(n_embd)
        head_dim = n_embd // n_head

        x = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)
        y = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)

        try:
            m_van = build_model(n_embd, n_head, args.n_layer, False, device, dtype,
                                args.vocab_size, args.block_size)
            n_params = m_van.get_num_params() / 1e6
            t_van = time_forward_backward(m_van, x, y, args.n_warmup, args.n_iter, device, ctx)
            del m_van
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            m_kv = build_model(n_embd, n_head, args.n_layer, True, device, dtype,
                               args.vocab_size, args.block_size)
            t_kv = time_forward_backward(m_kv, x, y, args.n_warmup, args.n_iter, device, ctx)
            del m_kv
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            overhead = (t_kv / t_van - 1) * 100
            results.append((n_embd, n_head, head_dim, n_params, t_van, t_kv, overhead))
            print(f"{n_embd:>7} {n_head:>7} {head_dim:>9} {n_params:>8.2f}M "
                  f"{t_van:>7.2f}ms {t_kv:>7.2f}ms {overhead:>+8.2f}%")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e)[:50]
            print(f"{n_embd:>7} SKIPPED: {msg}")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    if len(results) < 2:
        print("\nNot enough data points for trend analysis.")
        return

    print("\n" + "=" * 68)
    print("TREND ANALYSIS")
    print("=" * 68)
    smallest = results[0]
    largest = results[-1]
    print(f"  overhead at n_embd={smallest[0]:>4}: {smallest[6]:+.2f}%")
    print(f"  overhead at n_embd={largest[0]:>4}: {largest[6]:+.2f}%")
    ratio = largest[6] / smallest[6] if smallest[6] != 0 else float('nan')
    print(f"  ratio (large / small): {ratio:.2f}x")

    print("\nTheoretical baseline (FLOP-only):")
    print("  vanilla Linear ops per block ~ 16*E² (attn c_attn + c_proj + MLP fc + proj)")
    print("  K/V adds                     ~  4*E² (W_kvm + W_gm)")
    print("  Pure FLOP overhead ratio: 4/16 = 25%")
    print("  → If observed overhead >> 25%, kernel launch / memory bandwidth cost")
    print("    dominates at small sizes and should shrink toward 25% at large sizes.")
    print("  → If observed overhead is already close to 25%, further shrinkage is minimal.")

    print("\nInterpretation:")
    if largest[6] < smallest[6] * 0.5:
        print("  ✓ Overhead shrinks substantially at larger sizes — K/V likely to amortize.")
        print("  → Recommendation: Phase B (real training at n_embd~384-768) is worth $10-30.")
    elif largest[6] < smallest[6] * 0.8:
        print("  ~ Overhead shrinks moderately at larger sizes.")
        print("  → Recommendation: Phase B is a reasonable ~$15 gamble.")
    else:
        print("  ✗ Overhead does NOT shrink significantly with size.")
        print("  → K/V's wall-clock cost is inherent, not just small-scale artifact.")
        print("  → Recommendation: Do NOT spend on Phase B unless algorithmic gain")
        print("    also grows with scale (untested; separate question).")


if __name__ == "__main__":
    main()
