"""
Structural-understanding analysis: is K/V learning MORE MEANINGFUL patterns
than vanilla, or is its lower loss just uniform statistical smoothing?

Three tests:
  (1) Side-by-side text generation from the same prompts (subjective quality).
  (2) Per-character-type mean loss breakdown (where does K/V help vs vanilla).
  (3) Top-K positions where K/V wins most, with surrounding context, to see if
      the wins concentrate on structurally interesting positions (line breaks,
      character name starts, dialogue punctuation) or are spread randomly.

If K/V's win concentrates at structural positions -> mechanism learns structure.
If uniformly distributed -> just capacity smoothing.

Usage (Colab, after training):
    python analyze_understanding.py \\
        --out_van abl-42-VANILLA_n128_ref \\
        --out_kv  abl-42-KV_n112_full
"""

import argparse
import os
import pickle
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = dict(ckpt["model_args"])
    # Defensive: add defaults for any newer config keys that may not be in older checkpoints
    model_args.setdefault("m_gate_detached", False)
    cfg = GPTConfig(**model_args)
    model = GPT(cfg)
    sd = ckpt["model"]
    prefix = "_orig_mod."
    sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, cfg


def per_position_loss(model, x, y, ctx):
    """Per-position cross-entropy loss, shape (B, T).

    Note: must pass y so model.forward computes the FULL (B, T, V) logits;
    without targets, nanoGPT's inference optimization returns only the last
    position's logits (B, 1, V)."""
    with ctx, torch.no_grad():
        logits, _ = model(x, y)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T), reduction="none")
    return loss.view(B, T)


def char_type(c):
    if c == " ":       return "space"
    if c == "\n":      return "newline"
    if c in ".,!?;:":  return "punct"
    if c == "'":       return "apost"
    if c == "-":       return "dash"
    if "A" <= c <= "Z": return "upper"
    if "a" <= c <= "z": return "lower"
    if "0" <= c <= "9": return "digit"
    return "other"


def escape_ctx(s):
    """Make context safe for one-line printing."""
    return s.replace("\n", "\\n").replace("\t", "\\t")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_van", required=True, help="output dir containing vanilla ckpt.pt")
    p.add_argument("--out_kv", required=True, help="output dir containing K/V ckpt.pt")
    p.add_argument("--data_dir", default="data/shakespeare_char")
    p.add_argument("--n_batches", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--top_gap", type=int, default=20)
    p.add_argument("--sample_len", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    args = p.parse_args()

    device = args.device
    if device.startswith("cuda"):
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = nullcontext()

    # Vocabulary
    with open(os.path.join(args.data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    def encode(s): return [stoi[c] for c in s]
    def decode(ids): return "".join(itos[int(i)] for i in ids)

    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    # Models
    print(f"Loading vanilla: {args.out_van}")
    m_van, cfg_van = load_model(os.path.join(args.out_van, "ckpt.pt"), device)
    print(f"  params={m_van.get_num_params()/1e6:.2f}M  use_m_gate={cfg_van.use_m_gate}")
    print(f"Loading K/V:     {args.out_kv}")
    m_kv, cfg_kv = load_model(os.path.join(args.out_kv, "ckpt.pt"), device)
    print(f"  params={m_kv.get_num_params()/1e6:.2f}M  use_m_gate={cfg_kv.use_m_gate}")

    # ---- SECTION 1: Text generation from same prompts ----
    print("\n" + "=" * 78)
    print("SECTION 1  |  Text generation from same prompts (temperature=0.8, top_k=40)")
    print("=" * 78)
    prompts = ["ROMEO:\n", "First Citizen:\nWe are ", "JULIET: O Romeo, "]
    for prompt in prompts:
        prompt_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        for label, mdl in [("VANILLA", m_van), ("K/V-BIAS", m_kv)]:
            torch.manual_seed(42)  # same sampling stream for both models
            with ctx:
                out = mdl.generate(prompt_ids, args.sample_len,
                                    temperature=args.temperature, top_k=args.top_k)
            text = decode(out[0].tolist())
            print(f"\n--- {label}   prompt: {escape_ctx(prompt)!r} ---")
            print(text)
        print()

    # ---- SECTION 2 & 3: per-position loss analysis ----
    print("=" * 78)
    print(f"SECTION 2  |  Per-position loss on {args.n_batches} val batches "
          f"({args.n_batches*args.batch_size*args.block_size} positions total)")
    print("=" * 78)

    torch.manual_seed(1234)  # deterministic batch sampling
    all_gaps = []
    total_van, total_kv, n = 0.0, 0.0, 0
    per_type_van = defaultdict(list)
    per_type_kv = defaultdict(list)

    for _ in range(args.n_batches):
        ix = torch.randint(len(val_data) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([torch.from_numpy(val_data[i:i+args.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(val_data[i+1:i+1+args.block_size].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)

        loss_van = per_position_loss(m_van, x, y, ctx).float().cpu().numpy()
        loss_kv  = per_position_loss(m_kv,  x, y, ctx).float().cpu().numpy()
        total_van += loss_van.sum(); total_kv += loss_kv.sum(); n += loss_van.size

        y_np = y.cpu().numpy(); x_np = x.cpu().numpy()
        for bi in range(args.batch_size):
            for ti in range(args.block_size):
                target = itos[int(y_np[bi, ti])]
                gap = float(loss_van[bi, ti] - loss_kv[bi, ti])  # +ve = K/V better
                ctx_s = decode(x_np[bi, max(0, ti-24):ti+1].tolist())
                all_gaps.append((gap, target, ctx_s))
                per_type_van[char_type(target)].append(float(loss_van[bi, ti]))
                per_type_kv[char_type(target)].append(float(loss_kv[bi, ti]))

    print(f"\nOverall mean loss (nats):")
    print(f"  Vanilla: {total_van/n:.4f}")
    print(f"  K/V:     {total_kv/n:.4f}")
    print(f"  gap:     {(total_van-total_kv)/n:+.4f}   (positive = K/V better)")

    print(f"\nPer character-type breakdown:")
    print(f"  {'type':<10} {'count':>8} {'vanilla':>10} {'K/V':>10} {'gap':>10}  {'gap%':>7}")
    for t in sorted(per_type_van.keys(), key=lambda k: -len(per_type_van[k])):
        c = len(per_type_van[t])
        v = float(np.mean(per_type_van[t]))
        k = float(np.mean(per_type_kv[t]))
        pct = (v - k) / v * 100
        print(f"  {t:<10} {c:>8} {v:>10.4f} {k:>10.4f} {v-k:>+10.4f}  {pct:>+6.2f}%")

    print("\n" + "=" * 78)
    print(f"SECTION 3  |  Top {args.top_gap} positions where K/V beats vanilla most")
    print("=" * 78)
    print(f"  {'gap':>7}  {'target':>7}  context (last 25 chars before target)")
    print("  " + "-" * 74)
    for gap, ch, ctx_s in sorted(all_gaps, key=lambda z: -z[0])[:args.top_gap]:
        print(f"  {gap:>+7.3f}  {escape_ctx(repr(ch)):>7}  ...{escape_ctx(ctx_s)}")

    print(f"\n  Bottom {args.top_gap} (positions where K/V LOSES most vs vanilla):")
    print("  " + "-" * 74)
    for gap, ch, ctx_s in sorted(all_gaps, key=lambda z: z[0])[:args.top_gap]:
        print(f"  {gap:>+7.3f}  {escape_ctx(repr(ch)):>7}  ...{escape_ctx(ctx_s)}")

    print("\nInterpretation guide:")
    print("  - If K/V's per-type gaps are ROUGHLY UNIFORM across character types,")
    print("    the mechanism is mostly smoothing statistics — no clear structural role.")
    print("  - If K/V's gap is CONCENTRATED on: upper (character-name starts),")
    print("    newline (line breaks), punct (sentence structure), the mechanism")
    print("    is learning grammatical/structural patterns.")
    print("  - Compare 'top 20' contexts: do they cluster on syntactic pivots")
    print("    (start of a name after '\\n', word after ',' or '.', quote openings)")
    print("    or are they random?")


if __name__ == "__main__":
    main()
